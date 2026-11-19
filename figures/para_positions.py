# Token positions of the four paragraphs in results/qwen3-32b-grid-ci/G.json prompts (tokenizer only, no model), rebuilt as grid_ci_pod.py builds them.
# Usage (repo root): python figures/para_positions.py  -> results/qwen3-32b-grid-ci/para_positions.json (spans, and the 5 unmeasured tokens between paragraphs)
import json, os
os.environ["OMP_NUM_THREADS"]="2"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
R = json.load(open("results/qwen3-32b-grid-ci/G.json"))
INSTR = dict(none="", focus="Focus your attention on paragraph [P{k}].\n")
out = {}
for r in R:
    parts = ["Here are four paragraphs.\n\n"]; cspans = []; pos = len(parts[0])
    for i, p in enumerate(r["paras"]):
        lab = f"[P{i+1}] "; s = pos + len(lab); cspans.append((s, s + len(p)))
        seg = lab + p + "\n\n"; parts.append(seg); pos += len(seg)
    user = "".join(parts) + f"Unrelated question: what is {r['x']} + {r['y']}? Answer with just the number."
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = prompt.index(user); enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True); om = enc["offset_mapping"]
    sp = []
    for cs, ce in cspans:
        idx = [i for i, (a, b) in enumerate(om) if a >= cs + off and b <= ce + off and b > a]; sp.append((idx[0], idx[-1] + 1))
    toks = [[prompt[om[t][0]:om[t][1]] for t in range(lo, hi)] for lo, hi in sp]
    assert toks == r["tokens"], r["id"]
    gaps = [sp[i + 1][0] - sp[i][1] for i in range(3)]
    gtok = [[prompt[om[t][0]:om[t][1]] for t in range(sp[i][1], sp[i + 1][0])] for i in range(3)]
    out[r["id"]] = dict(spans=sp, gaps=gaps, gaptok=gtok)
json.dump(out, open("results/qwen3-32b-grid-ci/para_positions.json", "w"))
import collections
print(collections.Counter(g for v in out.values() for g in v["gaps"]))
print(list(out.values())[0]["gaptok"])
