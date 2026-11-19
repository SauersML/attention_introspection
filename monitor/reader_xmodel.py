# Cross-model reader: the "same model reading the transcript" arm of monitor.py (after|most, options in order),
# with the reader swapped for a model of another family. The reader text and question are built by monitor.py's own
# functions (user_content, when, question, optstr, reader_prefix_text: pulled from its source, unchanged); only the
# chat template is the reader model's. Score = probabilities of P1..P4 after the prefilled "P", as monitor.py reads().
# usage (GPU):  reader_xmodel.py read  MODEL OUT.json SHARD_DIR...   (shards hold contexts.json, gen1.json)
#       (score) reader_xmodel.py score OUT.json results/monitor_inorder/data/shard{0,1,2}  -> prints the table
import ast, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
NEED = {"user_content", "when", "question", "optstr", "reader_prefix_text", "DEF_SELF", "DEF_READ"}
src = ast.parse(open(os.path.join(HERE, "monitor.py")).read())
ns = {}
for n in src.body:
    name = n.name if isinstance(n, ast.FunctionDef) else (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
    if name in NEED: exec(compile(ast.Module([n], []), "monitor.py", "exec"), ns)
assert NEED <= set(ns), NEED - set(ns)
phase = sys.argv[1]

if phase == "read":
    MODEL, OUT, DIRS = sys.argv[2], sys.argv[3], sys.argv[4:]
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL)
    T = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    assert T("P3") == T("P") + T("3"), (T("P3"), T("P"), T("3"))
    DIG = [T(str(i)) for i in range(1, 5)]; assert all(len(d) == 1 for d in DIG), DIG; DIG = [d[0] for d in DIG]
    from transformers import AutoConfig, AutoModelForImageTextToText
    arch = (AutoConfig.from_pretrained(MODEL).architectures or [""])[0]   # Gemma 4 ships as an image-text model; text-only input here
    AM = AutoModelForImageTextToText if "ConditionalGeneration" in arch else AutoModelForCausalLM
    model = AM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    print("loaded", MODEL, arch, AM.__name__, flush=True)
    res = {}
    for d in DIRS:
        C = json.load(open(os.path.join(d, "contexts.json"))); G = json.load(open(os.path.join(d, "gen1.json")))
        for c in C:
            i = str(c["id"])
            text = ns["reader_prefix_text"](c, G[i]["ans_text"], "after") + ns["question"]("most", "reader", "after", c)
            s = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False) + "P"
            ids = T(s)
            with torch.no_grad():
                lg = torch.log_softmax(model(torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1].double(), -1)
            res[i] = dict(p=torch.softmax(lg[DIG], -1).tolist(), mass=float(lg[DIG].exp().sum()), ntok=len(ids))
            if len(res) == 1: res[i]["prompt"] = s
            print(i, np.round(res[i]["p"], 3), round(res[i]["mass"], 3), flush=True)
    json.dump(dict(model=MODEL, reads=res), open(OUT, "w"))

if phase == "score":
    OUT, DIRS = sys.argv[2], sys.argv[3:]
    from scipy.stats import rankdata
    X = json.load(open(OUT)); R = X["reads"]; H = {}
    for d in DIRS: H.update(json.load(open(os.path.join(d, "hf1.json"))))
    IDS = sorted(H, key=int); assert set(IDS) == set(R), (len(IDS), len(R))
    BAND = list(range(21, 54))   # as logprob.py: truth "norm", answer queries, mean over heads, layers 21-53
    def spear(a, b):
        ra, rb = rankdata(a), rankdata(b)
        if np.ptp(ra) == 0 or np.ptp(rb) == 0: return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(0)
    def boot(x, nb=10000):
        x = np.asarray(x, float); m = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
        return dict(mean=float(x.mean()), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)))
    NF = 200000
    def signflip(x):  # two-sided paired sign-flip test of mean(x) = 0, as the figure's p
        x = np.asarray(x, float); obs = abs(x.mean())
        b = sum(int((np.abs((np.random.default_rng(k).choice([-1.0, 1.0], (20000, len(x))) * x).mean(1)) >= obs - 1e-12).sum()) for k in range(10))
        return (b + 1) / (NF + 1)
    lp = lambda v: np.log(np.maximum(np.asarray(v, float), 1e-300))
    rows = {"self": [], "qwen_reader": [], "x_reader": []}
    for i in IDS:
        t = np.array(H[i]["truth"]["norm"], float)[BAND].mean(0)
        rows["self"].append(spear(lp(H[i]["rep"]["after|most|self"]), t))
        rows["qwen_reader"].append(spear(lp(H[i]["rep"]["after|most|reader"]), t))
        rows["x_reader"].append(spear(lp(R[i]["p"]), t))
    rows = {k: np.array(v) for k, v in rows.items()}
    out = dict(model=X["model"], n=len(IDS), mass_mean=float(np.mean([R[i]["mass"] for i in IDS])),
               **{k: dict(boot(v), p_vs_0=signflip(v)) for k, v in rows.items()},
               self_minus_x_reader=dict(boot(rows["self"] - rows["x_reader"]), p=signflip(rows["self"] - rows["x_reader"])),
               self_minus_qwen_reader=dict(boot(rows["self"] - rows["qwen_reader"]), p=signflip(rows["self"] - rows["qwen_reader"])),
               qwen_reader_minus_x_reader=dict(boot(rows["qwen_reader"] - rows["x_reader"]), p=signflip(rows["qwen_reader"] - rows["x_reader"])),
               perprompt={k: v.tolist() for k, v in rows.items()}, ids=IDS)
    json.dump(out, open(OUT.replace(".json", "_scored.json"), "w"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("perprompt", "ids")}, indent=1))
