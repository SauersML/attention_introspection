#!/bin/bash
# usage: run_priv.sh SCRIPT OUTDIR SHARD NSHARD [MODEL] [LIMIT]
set -e
mkdir -p "$2"
mkdir -p "${HF_HOME:-/workspace}"
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9)
echo "free GB at HF_HOME: $free"
[ "$free" -ge 80 ] || { echo "not enough disk ($free GB)"; exit 3; }
python3 "$1" --model "${5:-Qwen/Qwen3-32B}" --out "$2" --shard "$3" --nshard "$4" --limit "${6:-0}"
