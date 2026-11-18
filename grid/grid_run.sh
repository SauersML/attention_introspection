#!/bin/bash
set -e
mkdir -p "$2"
python3 "$1" --model Qwen/Qwen3-32B --n 100 --seed 11 --out "$2"
