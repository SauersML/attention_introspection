# Per-context numbers for the hero figure (head patching and head ablation) from heads.pt. CPU, seconds.
import json, torch, numpy as np
from figlib import boot_ci, signflip_p
d = torch.load("heads.pt", weights_only=False); R = d["R"]
Ls = list(range(27, 54))
def M_sel(att, a, b): return (att[Ls, b].log() - att[Ls, a].log()).mean().item()
out = {}
RAND_M = [f"rand_mover{i}" for i in range(4)]
for cond in ["focus", "ignore"]:
    rs = [r for r in R if r["kind"] == "kswap_head" and r["cond"] == cond]
    den = np.array([M_sel(r["donor"], r["a"], r["b"]) - M_sel(r["base"], r["a"], r["b"]) for r in rs])
    def frac(s): return np.array([M_sel(r["patched"][s], r["a"], r["b"]) - M_sel(r["base"], r["a"], r["b"]) for r in rs])
    mv = frac("movers"); rd = np.mean([frac(s) for s in RAND_M], 0)
    # ratio of means with bootstrap over contexts
    rng = np.random.default_rng(0); idx = rng.integers(0, len(den), (4000, len(den)))
    def rb(x): r = x[idx].mean(1) / den[idx].mean(1); return [x.mean() / den.mean(), np.quantile(r, .025), np.quantile(r, .975)]
    out[f"patch/{cond}"] = dict(movers=rb(mv), random=rb(rd), p=signflip_p(mv - rd), n=len(den))
    ab = [r for r in R if r["kind"] == "ablate"]
    def eff(s):  # % change of attention to the named paragraph, cond vs mention, both with heads s switched off
        m = np.array([r["att"][s][Ls, r["a"]].mean().item() for r in ab if r["cond"] == "mention"])
        c = np.array([r["att"][s][Ls, r["a"]].mean().item() for r in ab if r["cond"] == cond])
        return (c / m - 1) * 100
    e0, em = eff("none"), eff("movers"); er = np.mean([eff(s) for s in RAND_M], 0)
    out[f"ablate/{cond}"] = dict(intact=boot_ci(e0), movers=boot_ci(em), random=boot_ci(er),
                                 p_movers_vs_random=signflip_p(em - er), n=len(e0))
print(json.dumps(out, indent=1, default=float))
json.dump(out, open("hero_stats.json", "w"), default=float)
