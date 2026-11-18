#!/bin/bash
set -e
mkdir -p "$2"
mkdir -p "${HF_HOME:-/workspace}"
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9)
echo "free GB at HF_HOME: $free (model in cache: $(du -sh ${HF_HOME:-/workspace}/hub 2>/dev/null | cut -f1))"
[ "$free" -ge 70 ] || { echo "not enough disk ($free GB)"; exit 3; }
python3 "$1" --model Qwen/Qwen3-32B --n 100 --seed 2 --out "$2"
