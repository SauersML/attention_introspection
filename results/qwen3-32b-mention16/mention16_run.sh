#!/bin/bash
set -e
mkdir -p "$2"
DOSE_CONDS=0_none,1_mention,9_completely_ignore,1b_mention16 python3 "$1" --model Qwen/Qwen3-32B --n 150 --seed 7 --out "$2"
