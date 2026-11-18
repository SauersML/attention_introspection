#!/bin/bash
set -e
mkdir -p "$5"
python3 "$1" --model "$3" --n "$4" --out "$5"
python3 "$2" "$5"
