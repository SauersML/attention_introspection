#!/bin/bash
set -e
mkdir -p "$3"
python3 "$1" --model Qwen/Qwen3-32B --n 100 --seed 2 --out "$3"
python3 "$2" "$3"
python3 "$4" "$3"
