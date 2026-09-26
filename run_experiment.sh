#!/usr/bin/env bash
# Execute one experiment notebook headlessly.
#
#   ./run_experiment.sh similar          # full run, as reported in the paper
#   KVL_FAST=1 ./run_experiment.sh similar   # small smoke run, minutes not hours
#
# Results land in results/, figures in figures/. The default output tag
# reproduces the committed filenames (e.g. results/similar_similar.json).
set -euo pipefail

exp="${1:?usage: $0 <similar|depth|oracle|kappa|grid|xkvlora|multixkv|partial>}"
nb="notebooks/${exp}.ipynb"
[[ -f "$nb" ]] || { echo "no such notebook: $nb" >&2; exit 1; }

mkdir -p results figures

# Notebooks read either KVL_FAST or MXKV_FAST depending on the experiment;
# set both from one variable so callers need only KVL_FAST.
if [[ "${KVL_FAST:-0}" == "1" ]]; then
  export KVL_FAST=1 MXKV_FAST=1
  echo "[fast] reduced eval size; results are a smoke check, not paper numbers"
fi
export TOKENIZERS_PARALLELISM=false

python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output "${exp}.executed.ipynb" "$nb"

echo "done: $exp"
ls -la results/ | tail -n +2
