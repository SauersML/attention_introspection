#!/bin/bash
# usage: launch_notask.sh CODE.TAR OUTDIR SHARD NSHARD   (structure check: --notask)
set -e
W=$(mktemp -d); tar --no-same-owner -xf "$1" -C "$W"
EXTRA=--notask bash "$W/run.sh" "$W/told_inject_words.py" "$2" "$3" "$4"
