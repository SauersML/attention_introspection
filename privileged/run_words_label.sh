#!/bin/bash
# Label-vs-word test of the report-time boost (arm C only): for each target (word, label, both) run privileged_words.py --conly.
# usage: run_words_label.sh OUTDIR ORDER(inorder|shuffled) SHARD NSHARD   (run from the unpacked tar)
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$1" "${HF_HOME:-/workspace}"
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9); echo "free GB at HF_HOME: $free"
[ "$free" -ge 80 ] || { echo "not enough disk ($free GB)"; exit 3; }
python3 -c "import transformers,sys; sys.exit(0 if transformers.__version__=='5.18.0' else 1)" || pip install -q transformers==5.18.0
IO=""; [ "$2" = inorder ] && IO="--inorder"
for T in word label both; do
  mkdir -p "$1/$T"
  python3 "$D/privileged_words.py" $IO --conly --target $T --out "$1/$T" --shard "$3" --nshard "$4" 2>&1 | tee "$1/$T/run_$3.log"
done
