#!/bin/bash
# In-order rerun of monitor.py (report options listed P1, P2, P3, P4; labels were already in order); everything else as before.
# usage: run_inorder.sh OUTDIR SHARD NSHARDS PER_FAMILY   (run from the unpacked tar: monitor.py beside it)
set -e
D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$1" "${HF_HOME:-/workspace/hf-home}"
python3 -c "import transformers" || pip install -q transformers==5.18.0   # keep the env's version: vLLM is pinned to it
( while true; do touch "$1/BUSY"; sleep 60; done ) & TOUCHER=$!; trap "kill $TOUCHER 2>/dev/null" EXIT
export VLLM_ENABLE_V1_MULTIPROCESSING=0
for ph in gen1 hf1 gen2 hf2; do
  if [ -f "$1/$ph.done" ]; then echo "skip $ph"; continue; fi
  python3 "$D/monitor.py" --inorder --phase $ph --out "$1" --shard $2 --nshards $3 --per_family $4
  touch "$1/$ph.done"
done
echo ALLDONE
