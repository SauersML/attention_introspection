#!/bin/bash
# usage: run_analysis.sh analyze_monitor.py fig_monitor.py OUTDIR TITLE SHARD_DIR...
set -e
A=$1; F=$2; O=$3; TITLE=$4; shift 4
mkdir -p "$O"
python3 -m pip install -q matplotlib 2>&1 | tail -1 || true
python3 "$A" "$O" "$@"
python3 "$F" "$O/results.json" "$O/self_minus_reader.png" "$TITLE"
