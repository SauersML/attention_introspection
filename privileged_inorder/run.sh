#!/bin/bash
# usage: run.sh OUTDIR   (run from the unpacked tar)
set -e
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
python3 -c "import transformers,sys; sys.exit(0 if transformers.__version__=='5.18.0' else 1)" || pip install -q transformers==5.18.0
python3 -c "import datasets" || pip install -q datasets
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9); echo "free GB: $free"
python3 "$D/privileged_inorder.py" --out "$1" --shard 0 --nshard 1 2>&1 | tee "$1/run_0.log"
python3 "$D/analyze_inorder.py" "$1/results.json" "$1"/recs_0.jsonl -- "$D"/shuf/recs_*.jsonl 2>&1 | tee "$1/analyze.log"
