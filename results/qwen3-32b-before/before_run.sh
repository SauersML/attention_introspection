#!/bin/bash
set -e
mkdir -p "$2"
python3 "$1" --model Qwen/Qwen3-32B --n 150 --seed 7 --out "$2"
