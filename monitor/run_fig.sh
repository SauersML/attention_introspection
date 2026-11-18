#!/bin/bash
# usage: run_fig.sh FIG.py pal.py results.json OUT.png
set -e
mkdir -p "$(dirname "$4")"; cp "$2" "$(dirname "$1")/pal.py" 2>/dev/null || true
python3 -m ensurepip >/dev/null 2>&1 || true
python3 -m pip install -q matplotlib 2>&1 | tail -2
python3 "$1" "$3" "$4"
