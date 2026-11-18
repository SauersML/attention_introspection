#!/bin/bash
set -e
mkdir -p "$3"
mkdir -p "${HF_HOME:-/workspace}"
for d in /workspace "${HF_HOME:-/workspace}"; do df -h "$d"; done
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9)
echo "free GB at HF_HOME: $free (model already in cache: $(du -sh ${HF_HOME:-/workspace}/hub 2>/dev/null | cut -f1))"
[ "$free" -ge 70 ] || { echo "not enough disk ($free GB) before model load"; exit 3; }
python3 "$1" --model "$4" --n 150 --seed 7 --out "$3"
python3 "$2" "$3"
