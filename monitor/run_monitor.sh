#!/bin/bash
# usage: run_monitor.sh monitor.py OUTDIR SHARD NSHARDS PER_FAMILY [RESUME_DIR]
set -e
mkdir -p "$2" "${HF_HOME:-/workspace/hf-home}"
if [ -n "$6" ] && [ -d "$6" ]; then cp -n "$6"/*.json "$6"/*.done "$2"/ 2>/dev/null || true; fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for ph in gen1 hf1 gen2 hf2; do
  if [ -f "$2/$ph.done" ]; then echo "skip $ph"; continue; fi
  python3 "$1" --phase $ph --out "$2" --shard $3 --nshards $4 --per_family $5
  touch "$2/$ph.done"
done
echo ALLDONE
