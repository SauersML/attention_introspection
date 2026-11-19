#!/bin/bash
# usage: launch.sh CODE.TAR OUTDIR ARGS...
set -e
W=$(mktemp -d); tar --no-same-owner -xf "$1" -C "$W"; O="$2"; shift 2
bash "$W/run_words_inorder.sh" "$O" "$@"
