"""Check the committed result JSONs against the numbers reported in the paper.

No GPU, no model download, no network: this only re-reads results/*.json.
Exits non-zero if any reported value is not reproduced.

    python verify_results.py
"""
import json, math, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
load = lambda n: json.load(open(os.path.join(R, n)))

FLOOR, REFERENCE, N_EVAL = 0.180, 0.665, 200
rows, failures = [], 0


def check(section, label, got, want, tol=5e-3):
    global failures
    ok = got is not None and abs(got - want) <= tol
    failures += (not ok)
    rows.append((section, label, want, got, "ok" if ok else "MISMATCH"))


sim = load("similar_similar.json")
check("3.1", "noise at cosine 0.995 transfers", sim["armA"]["acc"]["cos0.995"], 0.005)
check("3.1", "genuine cache transfers", sim["armB"]["acc"]["a1.0"], 0.445)
check("3.1", "  ...at achieved cosine", sim["armB"]["achieved"]["a1.0"], 0.9949, tol=5e-4)
check("3.1", "floor anchor", sim["armA"]["acc"]["floor"], FLOOR)
check("3.1", "native-cache reference anchor", sim["armA"]["acc"]["ceiling"], REFERENCE)

# The inversion the section is named for: lower cosine, far better transfer.
ratio = sim["armB"]["acc"]["a1.0"] / sim["armA"]["acc"]["cos0.995"]
check("3.1", "genuine/noise transfer ratio (89x)", ratio, 89.0, tol=1.0)
assert sim["armB"]["achieved"]["a1.0"] < sim["armA"]["achieved"]["cos0.995"], \
    "the genuine cache should sit at a LOWER cosine than the noise condition"

kap = load("kappa_kappa.json")
check("4.1", "off/on readout gain ratio (logit)", kap["ratio_off_on"], 1.47, tol=0.02)
check("4.1", "off/on gain ratio (softmax output)", kap["ratio_out_off_on"], 1.48, tol=0.02)
check("4.1", "grouped-query group size G", kap["N_Q"] / kap["N_KV"], 6.0)
check("4.1", "layers", float(kap["N_LAYER"]), 28.0)

xl = load("xkvlora_xkvlora.json")["runs"]
check("5.2", "augmentation, zero trained params", xl["same:raw"]["xkv_shared"], 0.555)
check("5.2", "augmentation, query-side LoRA (qkv)", xl["same:qkv"]["xkv_shared"], 0.695)
check("5.2", "cross-family (GPT-Neo sender)", xl["cross:raw"]["xkv_shared"], 0.010)

# The paper reports 0.695 against the 0.665 reference as parity, not a win.
# z is the pooled two-proportion statistic, the convention the paper uses.
def z_two_prop(p1, p2, n=N_EVAL):
    p = (p1 + p2) / 2
    return (p1 - p2) / math.sqrt(2 * p * (1 - p) / n)

gap_items = (xl["same:qkv"]["xkv_shared"] - REFERENCE) * N_EVAL
check("5.2", "0.695-vs-reference gap, in items", gap_items, 6.0, tol=0.6)
check("5.2", "  ...as a z-score (parity)", z_two_prop(xl["same:qkv"]["xkv_shared"], REFERENCE), 0.64, tol=0.05)
check("5.2", "zero-param arm vs floor, z", z_two_prop(xl["same:raw"]["xkv_shared"], FLOOR), 7.8, tol=0.15)

mx = load("multixkv_multixkv.json")["runs"]
for n, want in [(1, 0.785), (2, 0.735), (4, 0.550), (8, 0.395)]:
    check("5.3", f"sender pool N={n} (trained qkv)", mx[f"qkv:N{n}"]["shared"], want)
for n in (1, 2, 4, 8):
    r = mx[f"qkv:N{n}"]
    rows.append(("5.3", f"  N={n} control-subtracted lift", None,
                 round(r["shared"] - max(r["ctrl_masked"], r["ctrl_distract"]), 3), "info"))

print(f"{'§':<5}{'claim':<40}{'paper':>8}{'shipped':>10}   status")
print("-" * 76)
for sec, label, want, got, status in rows:
    w = "  --  " if want is None else f"{want:.3f}"
    g = " n/a " if got is None else f"{got:.4f}"
    print(f"{sec:<5}{label:<40}{w:>8}{g:>10}   {status}")
print("-" * 76)
print(f"anchors: floor {FLOOR}   native-cache reference {REFERENCE}   n={N_EVAL}")
checked = [r for r in rows if r[4] != "info"]
print(f"{len(checked) - failures}/{len(checked)} reported values reproduced from results/")
if failures:
    print("\nMISMATCH above: the shipped JSON disagrees with the paper.", file=sys.stderr)
sys.exit(1 if failures else 0)
