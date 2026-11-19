#!/bin/bash
# usage: launch.sh CODE.TAR OUTDIR SHARD NSHARD
set -e
W=$(mktemp -d); tar -xf "$1" -C "$W"
bash "$W/run.sh" "$W/told_followup.py" "$2" "$3" "$4"
