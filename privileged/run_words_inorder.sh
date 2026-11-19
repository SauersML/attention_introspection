#!/bin/bash
# In-order rerun of privileged_words.py (labels [W1]..[W8] in slots 1..8, options in order); everything else as the original run.
# usage: run_words_inorder.sh OUTDIR SHARD NSHARD   (run from the unpacked tar: privileged_words.py and analyze_words.py beside it)
set -eo pipefail
D=$(cd "$(dirname "$0")" && pwd); mkdir -p "$1" "${HF_HOME:-/workspace}"
free=$(df -BG --output=avail "${HF_HOME:-/workspace}" | tail -1 | tr -dc 0-9); echo "free GB at HF_HOME: $free"
[ "$free" -ge 80 ] || { echo "not enough disk ($free GB)"; exit 3; }
python3 -c "import transformers,sys; sys.exit(0 if transformers.__version__=='5.18.0' else 1)" || pip install -q transformers==5.18.0
python3 "$D/privileged_words.py" --inorder --out "$1" --shard "$2" --nshard "$3" 2>&1 | tee "$1/run_$2.log"
