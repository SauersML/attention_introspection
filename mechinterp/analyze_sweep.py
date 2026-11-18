# Per-head effects and linear probes from sweep.pt.
import sys, json, torch
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/out"
d = torch.load(f"{OUT}/sweep.pt", weights_only=False)
att, lg, res, meta = d["att"].float(), d["logit"].float(), d["res"], d["meta"]
CONDS = ["none", "mention", "focus", "ignore"]
nC = att.shape[0] // 4; NL, H = att.shape[1], att.shape[3]
att = att.view(nC, 4, NL, 4, H, -1)  # ctx, cond, layer, k, head, span
lg = lg.view(nC, 4, NL, 4, H, -1)
tgt = att[..., :4].permute(0, 1, 2, 4, 3, 5)  # ctx cond layer head k span
T = torch.stack([tgt[:, :, :, :, i, i] for i in range(4)], -1)   # attention to the named paragraph [ctx cond L H k]
O = torch.stack([(tgt[:, :, :, :, i, :4].sum(-1) - tgt[:, :, :, :, i, i]) / 3 for i in range(4)], -1)  # mean of the other three
lgt = lg[..., :4].permute(0, 1, 2, 4, 3, 5)
LT = torch.stack([lgt[:, :, :, :, i, i] for i in range(4)], -1)
LO = torch.stack([(lgt[:, :, :, :, i, :4].sum(-1) - lgt[:, :, :, :, i, i]) / 3 for i in range(4)], -1)
r = {}
ci = {c: i for i, c in enumerate(CONDS)}
band = list(range(21, 54))
def boot(x, nb=2000):  # x: [ctx] -> mean, lo, hi over contexts
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, len(x), (nb, len(x)), generator=g)
    m = x[idx].mean(1); return [x.mean().item(), m.quantile(.025).item(), m.quantile(.975).item()]
for cond in ["focus", "ignore", "none"]:
    a, m = ci[cond], ci["mention"]
    dT = (T[:, a] - T[:, m]).mean(-1)  # [ctx, L, H] absolute change in attention per paragraph token
    dsel = ((T[:, a] - O[:, a]) - (T[:, m] - O[:, m])).mean(-1)
    dlog = ((LT[:, a] - LO[:, a]) - (LT[:, m] - LO[:, m])).mean(-1)  # change in pre-softmax score of Pk vs others
    base = T[:, m].mean(-1)
    md = dT.mean(0)  # [L, H]
    hm = md[band].mean(1).sum()  # total head-mean change summed over band layers
    flat = md.flatten(); order = flat.argsort(descending=(cond != "ignore"))
    cum = [(flat[order[:n]].sum() / md.sum()).item() for n in [1, 5, 10, 20, 50, 100, 200, 500]]
    top = [dict(L=int(i // H), h=int(i % H), dAtt=flat[i].item(), base=base.mean(0).flatten()[i].item(),
                rel=(flat[i] / base.mean(0).flatten()[i]).item(), dSel=dsel.mean(0).flatten()[i].item(),
                dLogit=dlog.mean(0).flatten()[i].item(), frac_ctx_same_sign=((dT.flatten(1)[:, i] > 0).float().mean().item()))
           for i in order[:30].tolist()]
    r[cond] = dict(top=top, cum_frac_of_total_change=dict(zip([1, 5, 10, 20, 50, 100, 200, 500], cum)),
                   per_layer_sum=md.sum(1).tolist(), per_layer_logit=dlog.mean(0).mean(1).tolist(),
                   headmean_band_rel=boot((T[:, a] - T[:, m]).mean(-1)[:, band].mean((1, 2)) / T[:, m].mean(-1)[:, band].mean((1, 2)) * 100))
    torch.save(dict(dT=md, dsel=dsel.mean(0), dlog=dlog.mean(0), base=base.mean(0)), f"{OUT}/heads_{cond}.pt")
# overlap of focus and ignore heads
f = torch.load(f"{OUT}/heads_focus.pt")["dT"].flatten(); g = torch.load(f"{OUT}/heads_ignore.pt")["dT"].flatten()
r["corr_focus_ignore_dAtt"] = torch.corrcoef(torch.stack([f, g]))[0, 1].item()
tf = set(f.argsort(descending=True)[:50].tolist()); tg = set(g.argsort()[:50].tolist())
r["overlap_top50_focus_up_ignore_down"] = len(tf & tg)
# probes: ridge one-vs-rest, 5-fold by context, predicting k
def probe(X, y, groups, lam=None):
    X = X.float(); n = X.shape[0]
    acc = []
    for f_ in range(5):
        te = (groups % 5) == f_; tr = ~te
        Xtr, Xte = X[tr], X[te]; mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-4
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Y = torch.nn.functional.one_hot(y[tr], 4).float() - .25
        K = Xtr @ Xtr.T; l = (lam or 1.0) * K.diagonal().mean()
        A = torch.linalg.solve(K + l * torch.eye(len(K)), Y)
        pred = (Xte @ Xtr.T @ A).argmax(1)
        acc.append((pred == y[te]).float())
    return torch.cat(acc).mean().item()
res = res.view(nC, 4, NL, 4, 4, -1).float()  # ctx cond L k pos d ; pos: digit, instr_end, last_prompt, answer_mean
y = torch.arange(4).repeat(nC); groups = torch.arange(nC).repeat_interleave(4)
P = {}
for cond in CONDS:
    for pi, pn in enumerate(["digit", "instr_end", "last_prompt", "answer_mean"]):
        if cond == "none" and pi < 2: continue
        P[f"{cond}/{pn}"] = [probe(res[:, ci[cond], L, :, pi].reshape(nC * 4, -1), y, groups) for L in range(0, NL, 2)]
r["probe_layers"] = list(range(0, NL, 2)); r["probe"] = P
json.dump(r, open(f"{OUT}/sweep_summary.json", "w"), indent=1)
for cond in ["focus", "ignore", "none"]:
    print(cond, "band rel", r[cond]["headmean_band_rel"], "cum", {k: round(v, 3) for k, v in r[cond]["cum_frac_of_total_change"].items()})
    for t in r[cond]["top"][:12]: print("  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in t.items()})
print("corr focus/ignore per-head dAtt", r["corr_focus_ignore_dAtt"], "overlap", r["overlap_top50_focus_up_ignore_down"])
for k, v in P.items(): print(k, " ".join(f"{x:.2f}" for x in v))
