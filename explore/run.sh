#!/bin/bash
# usage: run.sh NAME script args...  (runs on the pod in /workspace/x, background, log in out/NAME/log.txt)
cd /workspace/x; n=$1; shift; mkdir -p out/$n
export HF_HOME=/workspace/hf-home HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nohup /workspace/pyenv/2020f9fd6d35/bin/python "$@" > out/$n/log.txt 2>&1 &
echo $! > out/$n/pid
