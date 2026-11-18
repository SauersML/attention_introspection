import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {"n": len(R)}
ABL = ["intact", "early6", "early_rand", "late15", "late_rand"]
for ab in ABL:
    m = lambda r, c: r["m"][f"{c}|{ab}"]
    fs = [m(r, "focus/orig") - m(r, "focus/alt") for r in R]; is_ = [m(r, "ignore/alt") - m(r, "ignore/orig") for r in R]
    win = lambda c: boot([float(m(r, c) > 0) for r in R])
    out[ab] = dict(none=boot([m(r, "none") for r in R]), focus_swing=boot(fs), ignore_swing=boot(is_),
                   orig_wins={c: win(c) for c in ("none", "focus/orig", "focus/alt", "ignore/orig", "ignore/alt")})
    o = out[ab]; print(f"{ab:11s} none {o['none'][0]:+.2f}  focus swing {o['focus_swing'][0]:+.2f} [{o['focus_swing'][1]:+.2f},{o['focus_swing'][2]:+.2f}]  ignore swing {o['ignore_swing'][0]:+.2f} [{o['ignore_swing'][1]:+.2f},{o['ignore_swing'][2]:+.2f}]  orig>alt: " + " ".join(f"{c} {v[0]:.2f}" for c, v in o["orig_wins"].items()))
for ab in ABL[1:]:
    for w, (c1, c2) in (("focus", ("focus/orig", "focus/alt")), ("ignore", ("ignore/alt", "ignore/orig"))):
        d = [(r["m"][f"{c1}|{ab}"] - r["m"][f"{c2}|{ab}"]) - (r["m"][f"{c1}|intact"] - r["m"][f"{c2}|intact"]) for r in R]
        out[f"delta/{ab}/{w}"] = boot(d) + (signp(d),)
        print(f"  {ab} change in {w} swing {out[f'delta/{ab}/{w}'][0]:+.2f} [{out[f'delta/{ab}/{w}'][1]:+.2f},{out[f'delta/{ab}/{w}'][2]:+.2f}] p={out[f'delta/{ab}/{w}'][3]:.4f}")
def selL(r, c, ab, L):
    a = r["att"][f"{c}|{ab}"].float()[L].mean(0); return np.log((a[5] / r["ntok"][c][5]).item() / (a[4] / r["ntok"][c][4]).item())
LL = list(range(45, 63))
for ab in ("intact", "early6", "early_rand"):
    f = [np.mean([selL(r, "focus/alt", ab, L) - selL(r, "focus/orig", ab, L) for L in LL]) for r in R]
    g = [np.mean([selL(r, "ignore/orig", ab, L) - selL(r, "ignore/alt", ab, L) for L in LL]) for r in R]
    out[f"latesel/{ab}"] = dict(focus=boot(f), ignore=boot(g)); print(f"late selection (layers 45-62, nats) {ab}: focus {out[f'latesel/{ab}']['focus']}, ignore {out[f'latesel/{ab}']['ignore']}")
json.dump(out, open(sys.argv[2], "w"), indent=1)
