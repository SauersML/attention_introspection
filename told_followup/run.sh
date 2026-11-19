#!/bin/bash
# usage: run.sh TOLD_FOLLOWUP.PY OUTDIR SHARD NSHARD [MODEL] [LIMIT]
set -e
mkdir -p "$2"
python3 -c "import transformers,sys; sys.exit(0 if transformers.__version__=='5.18.0' else 1)" || pip install -q transformers==5.18.0
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9); echo "free GB: $free"
python3 "$1" --model "${5:-Qwen/Qwen3-32B}" --out "$2" --shard "$3" --nshard "$4" --limit "${6:-0}"
