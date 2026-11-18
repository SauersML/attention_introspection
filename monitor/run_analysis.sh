#!/bin/bash
# usage: run_analysis.sh analyze_monitor.py fig_monitor.py pal.py OUTDIR TITLE SHARD_DIR...
set -e
A=$1; F=$2; PAL=$3; O=$4; TITLE=$5; shift 5
mkdir -p "$O"
cp "$PAL" "$(dirname "$F")/pal.py" 2>/dev/null || true
python3 -m ensurepip >/dev/null 2>&1 || true
python3 -m pip install -q matplotlib 2>&1 | tail -2
python3 "$A" "$O" "$@"
python3 "$F" "$O/results.json" "$O/self_minus_reader.png" "$TITLE"
