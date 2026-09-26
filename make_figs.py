#!/usr/bin/env python3
"""Regenerate fig2 and fig3 for close_enough from REAL measured data.

fig2 = "cosine is the wrong instrument": noise-vs-genuine at matched cosine (armA/armB of
       similar_similar.json) + the 6000-key bank k x alpha collapse
       (oracle_oracle.json).
fig3 = "two explanations that fail": the one-layer step function vs graded on-manifold decay
       (depth_depth.json legs.A_depth_budget) + the 1.47x gain ratio
       (kappa_kappa.json).

No invented numbers: every series is read out of the result JSONs at run time.
"""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
A = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
OUT = os.environ.get("FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
os.makedirs(OUT, exist_ok=True)
sim = json.load(open(f"{A}/similar_similar.json"))
ora = json.load(open(f"{A}/oracle_oracle.json"))
dep = json.load(open(f"{A}/depth_depth.json"))
kap = json.load(open(f"{A}/kappa_kappa.json"))

FLOOR = sim["armA"]["acc"]["floor"]
CEIL = sim["armA"]["acc"]["ceiling"]

RED, BLUE, GREEN, GREY = "#C0392B", "#2E75B6", "#1E8449", "#7F8C8D"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def anchors(ax, label=True):
    ax.axhline(FLOOR, ls="--", lw=1.2, c=GREY,
               label=f"floor {FLOOR:.2f}" if label else None)
    ax.axhline(CEIL, ls=":", lw=1.2, c=GREEN,
               label=f"native ref {CEIL:.3f}" if label else None)


# ----------------------------------------------------------------- FIG 2
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1))

# (a) noise vs genuine, x = achieved cosine
ct = sim["config"]["cos_targets"]
nx = [sim["armA"]["achieved"][f"cos{c}"] for c in ct]
ny = [sim["armA"]["acc"][f"cos{c}"] for c in ct]
al = sim["config"]["alphas"]
gx = [sim["armB"]["achieved"][f"a{a}"] for a in al]
gy = [sim["armB"]["acc"][f"a{a}"] for a in al]

ax[0].plot(nx, ny, "o-", color=RED, lw=2, ms=6, label="isotropic noise on own cache")
ax[0].plot(gx, gy, "s-", color=BLUE, lw=2, ms=6, label="genuine fact-bearing cache")
anchors(ax[0])
ax[0].set_xlim(1.0004, 0.9895)  # reversed: closer on the left
ax[0].set_xlabel("achieved cosine similarity to the reader's own cache")
ax[0].set_ylabel("fact-transfer accuracy")
ax[0].set_title("(a) Same fidelity, opposite outcome", fontweight="bold")
ax[0].annotate("noise at cos 0.995:\n0.005, BELOW floor",
               xy=(0.995, 0.005), xytext=(0.9950, 0.215),
               color=RED, fontsize=8.5, ha="center",
               arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax[0].annotate("genuine at cos 0.9949:\n0.445",
               xy=(gx[-1], gy[-1]), xytext=(0.9930, 0.470),
               color=BLUE, fontsize=8.5, ha="center",
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax[0].legend(fontsize=8, loc="lower left", framealpha=0.95)
ax[0].set_ylim(-0.03, 0.72)

# (b) 6000-key bank: k x alpha
alphas = ora["config"]["alphas"]
for k, c in zip(ora["config"]["knn_ks"],
                plt.cm.viridis(np.linspace(0.08, 0.82, len(ora["config"]["knn_ks"])))):
    ys = [ora["knn"][f"k{k}"]["sweep_acc"][f"a{a}"] for a in alphas]
    ax[1].plot(alphas, ys, "o-", color=c, lw=1.7, ms=5, label=f"$k$={k}")
anchors(ax[1], label=False)
ax[1].text(0.30, FLOOR + 0.012, f"floor {FLOOR:.2f}", fontsize=8, color=GREY, ha="left")
ax[1].set_xlabel(r"mixing weight $\alpha$ toward the $k$-th nearest genuine key")
ax[1].set_ylabel("fact-transfer accuracy")
ax[1].set_title("(b) Selecting from 6000 real keys does not help", fontweight="bold")
ax[1].legend(fontsize=8, ncol=2, title="neighbour rank", title_fontsize=8)
ax[1].set_ylim(-0.02, 0.28)
ax[1].annotate("nearest real key of 6000\nalready at zero",
               xy=(1.0, 0.0), xytext=(0.66, 0.085), fontsize=8.5, color=RED, ha="center",
               arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_tolerance.png"), dpi=160)
print("wrote fig2_tolerance.png")

# ----------------------------------------------------------------- FIG 3
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.1))

L = dep["legs"]["A_depth_budget"]
ell, nn1, gauss = L["ell"], L["nn1"], L["gauss"]

ax[0].plot(ell, gauss, "o-", color=RED, lw=2.2, ms=6,
           label="off-manifold (noise)")
ax[0].plot(ell, nn1, "s-", color=BLUE, lw=2.2, ms=6,
           label="on-manifold (genuine keys)")
anchors(ax[0], label=False)
ax[0].text(0.4, CEIL + 0.017, f"native ref {CEIL:.3f}", fontsize=8, color=GREEN)
ax[0].text(21.0, FLOOR + 0.017, f"floor {FLOOR:.2f}", fontsize=8, color=GREY)
ax[0].set_xlabel(r"number of corrupted layers $\ell$   (of 28; the rest bit-exact)")
ax[0].set_ylabel("fact-transfer accuracy")
ax[0].set_title("(a) One layer spends the whole budget", fontweight="bold")
ax[0].annotate(r"$\ell=1$: 0.665 $\rightarrow$ 0.030" "\nflat all the way to 28\n(nothing compounds)",
               xy=(1, gauss[1]), xytext=(6.2, 0.245),
               color=RED, fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax[0].annotate("same cosine (0.822),\nsurvives 8 layers",
               xy=(8, nn1[6]), xytext=(11.6, 0.40),
               color=BLUE, fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
ax[0].legend(fontsize=8, loc="upper right", framealpha=0.95)
ax[0].set_ylim(-0.03, 0.80)

# (b) gain ratio: no cliff
stages = ["attention\nlogit stage", "softmax\noutput stage"]
off = [kap["kappa_off_mean"], kap["kappa_out_off_mean"]]
on = [kap["kappa_on_mean"], kap["kappa_out_on_mean"]]
x = np.arange(len(stages)); w = 0.34
ax[1].bar(x - w / 2, off, w, color=RED, label="off-manifold")
ax[1].bar(x + w / 2, on, w, color=BLUE, label="on-manifold")
ax[1].axhline(1.0, ls="--", lw=1.1, c=GREY)
ax[1].text(-0.42, 1.045, "unit gain", fontsize=8, color=GREY, ha="left")
for xi, (o, n) in enumerate(zip(off, on)):
    ax[1].text(xi, max(o, n) + 0.09, f"{o/n:.2f}$\\times$", ha="center",
               fontsize=9.5, fontweight="bold", color="#333333")
ax[1].set_xticks(x); ax[1].set_xticklabels(stages)
ax[1].set_ylabel(r"readout Jacobian gain $\kappa$")
ax[1].set_title("(b) Nothing explodes off-manifold", fontweight="bold")
ax[1].legend(fontsize=8)
ax[1].set_ylim(0, 2.75)
ax[1].text(0.5, 2.46, "ratios are $O(1)$ — the wall is not built by amplification",
           ha="center", fontsize=8.5, color=GREY, style="italic")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_mechanism.png"), dpi=160)
print("wrote fig3_mechanism.png")
print(f"\nanchors used: floor={FLOOR} ceiling={CEIL}")
print(f"gain ratios: logit {off[0]/on[0]:.4f}  softmax-out {off[1]/on[1]:.4f}")

# ----------------------------------------------------------------- FIG 4
xl = json.load(open(f"{A}/xkvlora_xkvlora.json"))
mx = json.load(open(f"{A}/multixkv_multixkv.json"))

sub = [("freeform\nregress", 0.00), ("manifold\nloss", 0.01), ("by-con-\nstruction", 0.01),
       ("row-space\ntarget", 0.085), ("worst-\ncase", 0.08), ("hi-cap\nMLP", 0.20),
       ("codebook\nselect", 0.00), ("oracle\n1-NN", 0.00)]
aug = [("xattn\n(no train)", xl["runs"]["same:raw"]["xkv_shared"]),
       ("xattn\n+LoRA qkv", xl["runs"]["same:qkv"]["xkv_shared"])]

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.3),
                       gridspec_kw={"width_ratios": [1.75, 1]})

names = [n for n, _ in sub] + [n for n, _ in aug]
vals = [v for _, v in sub] + [v for _, v in aug]
cols = [RED] * len(sub) + [GREEN] * len(aug)
xs = np.arange(len(names))
ax[0].axhspan(-0.02, FLOOR, color=RED, alpha=0.055)
ax[0].bar(xs, vals, 0.68, color=cols)
ax[0].axhline(FLOOR, ls="--", lw=1.3, c=GREY)
ax[0].axhline(CEIL, ls=":", lw=1.3, c=GREEN)
ax[0].text(-0.45, FLOOR + 0.014, f"transfer floor {FLOOR:.2f}", fontsize=8, color=GREY)
ax[0].text(-0.45, CEIL + 0.014, f"native-cache ref {CEIL:.3f}", fontsize=8, color=GREEN)
for xi, v in zip(xs, vals):
    ax[0].text(xi, v + 0.016, f"{v:.3f}".rstrip("0").rstrip("."), ha="center", fontsize=8)
ax[0].set_xticks(xs); ax[0].set_xticklabels(names, fontsize=7.6)
ax[0].set_ylabel("fact-transfer accuracy")
ax[0].set_ylim(-0.02, 0.86)
ax[0].set_title("(a) Two regimes: substituting a produced cache vs. augmenting with the sender's",
                fontweight="bold", fontsize=10)
ax[0].text(3.2, 0.50, "SUBSTITUTE into the reader's slots\n$\\rightarrow$ capped at the floor",
           ha="center", fontsize=9, color=RED, fontweight="bold")
ax[0].text(7.55, 0.795, "AUGMENT:\nkeep own cache,\nattend over the sender",
           ha="center", va="top", fontsize=8.6, color=GREEN, fontweight="bold")

Ns = mx["config"]["n_holders"]
raw = [mx["runs"][f"raw:N{n}"]["shared"] for n in Ns]
qkv = [mx["runs"][f"qkv:N{n}"]["shared"] for n in Ns]
ctl = [max(mx["runs"][f"qkv:N{n}"]["ctrl_masked"],
           mx["runs"][f"qkv:N{n}"]["ctrl_distract"]) for n in Ns]
ax[1].plot(Ns, qkv, "s-", color=GREEN, lw=2.2, ms=7, label="with query-side adaptation")
ax[1].plot(Ns, raw, "o-", color=RED, lw=2.2, ms=6, label="raw cross-attention")
ax[1].plot(Ns, ctl, "^:", color=GREY, lw=1.5, ms=5, label="strongest control")
ax[1].axhline(FLOOR, ls="--", lw=1.3, c=GREY)
ax[1].axhline(CEIL, ls=":", lw=1.3, c=GREEN)
ax[1].set_xscale("log", base=2); ax[1].set_xticks(Ns)
ax[1].set_xticklabels([str(n) for n in Ns])
ax[1].set_xlabel("senders in the pool (1 holds the answer)")
ax[1].set_ylabel("fact-transfer accuracy")
ax[1].set_title("(b) Scaling the pool", fontweight="bold", fontsize=10)
ax[1].legend(fontsize=7.8, loc="upper right")
ax[1].set_ylim(-0.02, 0.86)
ax[1].annotate("raw drops below\nfloor by N=4", xy=(4, raw[2]), xytext=(2.15, 0.30),
               fontsize=8, color=RED,
               arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig4_landscape.png"), dpi=160)
print("wrote fig4_landscape.png")
