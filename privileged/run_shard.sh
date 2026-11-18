#!/bin/bash
# usage: run_shard.sh RUN_PRIV.SH PRIVILEGED_WORDS.PY ANALYZE_WORDS.PY OUTDIR SHARD NSHARD [MODEL] [LIMIT]
set -e
bash "$1" "$2" "$4" "$5" "$6" "${7:-Qwen/Qwen3-32B}" "${8:-0}"
python3 "$3" "$4" "$4"/recs_*.jsonl > "$4/analyze_shard.log" 2>&1 || echo "analysis failed"
cat "$4/analyze_shard.log"
