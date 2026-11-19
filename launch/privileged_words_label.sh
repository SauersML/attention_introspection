#!/bin/bash
# usage: launch.sh CODE.TAR OUTDIR ARGS...   (pod side; ARGS go to run_words_label.sh: ORDER SHARD NSHARD)
set -e
W=$(mktemp -d); tar --no-same-owner -xf "$1" -C "$W"; O="$2"; shift 2
bash "$W/run_words_label.sh" "$O" "$@"
