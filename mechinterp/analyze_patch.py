# Transfer fractions from patch.pt and heads.pt (run where the .pt files are; CPU only).
import sys, json, torch
OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/mi/out"
g = torch.Generator().manual_seed(0)
def ratio_boot(num, den, nb=4000):
    idx = torch.randint(0, len(num), (nb, len(num)), generator=g)
    r = num[idx].mean(1) / den[idx].mean(1)
    return [(num.mean() / den.mean()).item(), r.quantile(.025).item(), r.quantile(.975).item()]
def M_sel(att, a, b, Ls):  # att [NL, 4] -> mean over layers of log(att_b/att_a)
    return (att[Ls, b].log() - att[Ls, a].log()).mean()
def M_tgt(att, a, Ls):
    o = [i for i in range(4) if i != a]
    return (att[Ls, a].log() - att[Ls][:, o].mean(-1).log()).mean()
out = {}
try:
    d = torch.load(f"{OUT}/patch.pt", weights_only=False); R, LAYERS = d["R"], d["LAYERS"]
    for kind, conds in [("kswap", ["focus", "ignore"]), ("wording", ["focus", "ignore"])]:
        for cond in conds:
            rs = [r for r in R if r["kind"] == kind and r["cond"] == cond]
            sets = sorted(set(s for s, _ in rs[0]["patched"]))
            res = {}
            for s in sets:
                res[s] = {}
                for L in LAYERS:
                    Ls = list(range(max(L + 1, 21), 54))
                    if not Ls: continue
                    f = (lambda at, r: M_sel(at, r["a"], r["b"], Ls)) if kind == "kswap" else (lambda at, r: M_tgt(at, r["a"], Ls))
                    num = torch.stack([f(r["patched"][(s, L)], r) - f(r["base"], r) for r in rs])
                    den = torch.stack([f(r["donor"], r) - f(r["base"], r) for r in rs])
                    res[s][L] = ratio_boot(num, den) + [den.mean().item()]
            out[f"{kind}/{cond}"] = res
            print(kind, cond, "n", len(rs))
            for s in sets: print("  ", s, " ".join(f"{L}:{v[0]:+.2f}" for L, v in res[s].items()))
except FileNotFoundError: pass
try:
    d = torch.load(f"{OUT}/heads.pt", weights_only=False); R = d["R"]; out["SETS"] = {k: v for k, v in d["SETS"].items()}
    Ls = list(range(27, 54))
    for cond in ["focus", "ignore"]:
        rs = [r for r in R if r["kind"] == "kswap_head" and r["cond"] == cond]
        den = torch.stack([M_sel(r["donor"], r["a"], r["b"], Ls) - M_sel(r["base"], r["a"], r["b"], Ls) for r in rs])
        res = {s: ratio_boot(torch.stack([M_sel(r["patched"][s], r["a"], r["b"], Ls) - M_sel(r["base"], r["a"], r["b"], Ls) for r in rs]), den) for s in rs[0]["patched"]}
        out[f"headpatch/{cond}"] = res
        print("headpatch", cond, {k: [round(x, 3) for x in v] for k, v in res.items()})
    # ablation: effect of focus/ignore vs mention on attention to the named paragraph (layers 27-53 head-mean, relative %)
    for cond in ["focus", "ignore"]:
        res = {}
        for s in R[[r["kind"] for r in R].index("ablate")]["att"]:
            m = torch.stack([r["att"][s][Ls, r["a"]].mean() for r in R if r["kind"] == "ablate" and r["cond"] == "mention"])
            c = torch.stack([r["att"][s][Ls, r["a"]].mean() for r in R if r["kind"] == "ablate" and r["cond"] == cond])
            x = (c / m - 1) * 100
            idx = torch.randint(0, len(x), (4000, len(x)), generator=g); bm = x[idx].mean(1)
            res[s] = [x.mean().item(), bm.quantile(.025).item(), bm.quantile(.975).item()]
        out[f"ablate/{cond}"] = res
        print("ablate", cond, {k: [round(v, 1) for v in vv] for k, vv in res.items()})
    # label binding
    for cond in ["mention", "focus", "ignore"]:
        rs = [r for r in R if r["kind"] == "labels" and r["cond"] == cond]
        lab, posn, oth = [], [], []
        for r in rs:
            for k in range(4):
                il = r["perm"].index(k); ip = k
                at = r["att"][k][21:54].mean(0)
                lab.append(at[il]); posn.append(at[ip]); oth.append((at.sum() - at[il] - at[ip]) / 2)
        lab, posn, oth = map(torch.stack, (lab, posn, oth))
        out[f"labels/{cond}"] = dict(label=lab.mean().item(), position=posn.mean().item(), other=oth.mean().item(), lab=lab.tolist(), pos=posn.tolist(), oth=oth.tolist())
        print("labels", cond, "label-matched", lab.mean().item(), "position-matched", posn.mean().item(), "other", oth.mean().item())
except FileNotFoundError: pass
json.dump(out, open(f"{OUT}/patch_summary.json", "w"), default=str)
