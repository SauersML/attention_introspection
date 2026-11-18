# Head-level causal tests.
# A  k-swap head patching: copy chosen heads' outputs (positions after the instruction line) from the run naming
#    [Pb] into the run naming [Pa]; does attention follow to Pb?
# B  zero-ablation of the same heads (positions after the instruction line) in focus and mention runs.
# C  label binding: paragraphs printed with permuted labels; does "focus on [Pk]" follow the label or the position?
import sys, os, random, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import *
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/out"
tok, model = load(); NL = 64; H = 64
ctxs = contexts(tok)[:N]
MOVERS = [(20, 19), (20, 29), (23, 61), (25, 28), (25, 30), (26, 5)]  # top answer->instruction-digit heads (focus), layers < 28
hf = torch.load(f"{OUT}/heads_focus.pt")["dT"]
READERS = [(int(i) // H, int(i) % H) for i in hf.flatten().argsort(descending=True)[:10]]
rng = random.Random(0)
def rand_heads(n, lo, hi, excl):
    out = set()
    while len(out) < n:
        h = (rng.randrange(lo, hi + 1), rng.randrange(H))
        if h not in excl: out.add(h)
    return sorted(out)
SETS = {"movers": MOVERS, "readers": READERS}
for m in MOVERS: SETS[f"mover{m[0]}.{m[1]}"] = [m]
for r in range(4): SETS[f"rand_mover{r}"] = rand_heads(6, 20, 26, set(MOVERS))
for r in range(2): SETS[f"rand_reader{r}"] = rand_heads(10, 24, 29, set(READERS))
log("sets", SETS)

def run(ids, rows, spans, B=1, hp=None, hz=None, hcache=None):
    reset(); S.update(rec=True, rows=rows, spans=spans, headpatch=hp, headzero=hz, headcache=hcache)
    with torch.no_grad():
        model(torch.tensor([ids] * B, device="cuda"), use_cache=False, logits_to_keep=1)
    return torch.stack([S["store"][L][:, :, :4].float().mean(1) for L in range(NL)], 1)  # [B, NL, 4]

def group(specs):  # specs: list of head lists -> {L: [(b, heads)]}
    g = {}
    for b, hs in enumerate(specs):
        for L in sorted(set(l for l, _ in hs)):
            g.setdefault(L, []).append((b, [h for l, h in hs if l == L]))
    return g

R = []
names = list(SETS)
for n, c in enumerate(ctxs):
    ans = generate(tok, model, build(tok, c, "none", c["k"])[0])
    a = c["k"]; b = (a + 1 + n % 3) % 4
    for cond in ["focus", "ignore"]:
        idsA, infA = build(tok, c, cond, a); idsB, _ = build(tok, c, cond, b)
        idsA += ans; idsB += ans; T = len(idsA); rows = list(range(T - len(ans), T)); spans = infA["paras"]
        pos = list(range(infA["after"], T))
        base = run(idsA, rows, spans)[0]
        donor = run(idsB, rows, spans, hcache={L: pos for L in range(NL)})[0]
        hc = {L: v[0] for L, v in S["hc"].items()}
        out = {}
        for i in range(0, len(names), 8):
            chunk = names[i:i + 8]
            g = group([SETS[s] for s in chunk])
            hp = {L: [(bb, hs, pos, hc[L][hs]) for bb, hs in v] for L, v in g.items()}
            att = run(idsA, rows, spans, B=len(chunk), hp=hp)
            out.update({s: att[j] for j, s in enumerate(chunk)})
        R.append(dict(kind="kswap_head", id=c["id"], cond=cond, a=a, b=b, base=base, donor=donor, patched=out))
    # B ablation
    abl = ["none", "movers", "rand_mover0", "rand_mover1", "rand_mover2", "rand_mover3", "readers", "rand_reader0", "rand_reader1"]
    for cond in ["mention", "focus", "ignore"]:
        ids, inf = build(tok, c, cond, a); ids += ans; T = len(ids); rows = list(range(T - len(ans), T))
        pos = list(range(inf["after"], T))
        g = group([SETS[s] if s != "none" else [] for s in abl])
        hz = {L: [(bb, hs, pos) for bb, hs in v] for L, v in g.items()}
        att = run(ids, rows, inf["paras"], B=len(abl), hz=hz)
        R.append(dict(kind="ablate", id=c["id"], cond=cond, a=a, att={s: att[j] for j, s in enumerate(abl)}))
    # C label binding: a derangement of the labels
    perm = [1, 2, 3, 0] if n % 2 == 0 else [3, 0, 1, 2]
    for cond in ["mention", "focus", "ignore"]:
        bt = [build(tok, c, cond, k, perm) for k in range(4)]
        ids = [x[0] + ans for x in bt]; inf = bt[0][1]; T = len(ids[0]); rows = list(range(T - len(ans), T))
        reset(); S.update(rec=True, rows=rows, spans=inf["paras"])
        with torch.no_grad(): model(torch.tensor(ids, device="cuda"), use_cache=False, logits_to_keep=1)
        att = torch.stack([S["store"][L][:, :, :4].float().mean(1) for L in range(NL)], 1)  # [k, NL, para position]
        R.append(dict(kind="labels", id=c["id"], cond=cond, perm=perm, att=att))
    log("ctx", n + 1)
    if (n + 1) % 10 == 0: torch.save(dict(R=R, SETS=SETS), f"{OUT}/heads.pt")
torch.save(dict(R=R, SETS=SETS), f"{OUT}/heads.pt")
log("done")
