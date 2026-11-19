# grid_focus_L28.py with only its first pick (the "focus" context just below the pool mean): one typical question.
# Usage: python grid_focus_one.py results/qwen3-32b-grid-ci/G.json  (same data as grid_focus_L28.py)
# Token-level grid: Qwen3-32B, arithmetic task, 100 new contexts (seed 11) x 13 versions (grid_pod.py).
# Picks, for "focus" and for "ignore", the context just below and just above the pool's mean effect,
# and draws all four paragraphs' tokens coloured by % change vs the matching "only mentioned" run.
import json, html, os, subprocess, sys, numpy as np
from matplotlib import colors
G = sys.argv[1]
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
CHROME = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # shared figures/pal.py
from pal import PURPLE, RED, INK, DIV
R = json.load(open(G)); byid = {r["id"]: r for r in R}

# ---- checks
rel = max(abs(np.mean(v["tok_L28"][m]) / v["span_L28"][m] - 1) for r in R for v in r["vers"].values() for m in range(4))
nfree = sum(v["free_same"] for r in R for k, v in r["vers"].items() if k != "none")
ntot = sum(1 for r in R for k in r["vers"] if k != "none")
acc = sum(r["ans_none"].strip() == str(r["x"] + r["y"]) for r in R)
print(f"max relative diff, paragraph mean from tokens vs span mean: {rel:.2e}")
print(f"free answers identical to no-instruction answer: {nfree}/{ntot}; no-instruction correct {acc}/{len(R)}")

# ---- effects and picks
def eff(r, cond): return float(np.mean([100 * (r["vers"][f"{cond}{k}"]["span_L28"][k - 1] / r["vers"][f"mention{k}"]["span_L28"][k - 1] - 1) for k in range(1, 5)]))
E = {c: {r["id"]: eff(r, c) for r in R} for c in ("focus", "ignore")}
MEAN = {c: float(np.mean(list(E[c].values()))) for c in E}
print("pool means", {c: round(MEAN[c], 2) for c in MEAN})
picks, used = [], set()
for c in ("focus",):
    for side in ("below", "above"):
        cand = sorted((i for i in E[c] if (E[c][i] < MEAN[c] if side == "below" else E[c][i] > MEAN[c])), key=lambda i: abs(E[c][i] - MEAN[c]))
        i = next(i for i in cand if i not in used); used.add(i); picks.append((c, side, i))
        print(c, side, "context", i, f"effect {E[c][i]:+.2f}%")
picks = picks[:1]  # the top example only

# ---- tiles: every measured token of all four paragraphs as a chip (width follows the token's length), coloured with the
# shared pal.pct_colour by its attention under "focus" / under the matching "only mentioned" prompt; one blue chat bubble per named paragraph
FIG = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, FIG)
from pal import pct_colour
from hero_cols import pct, CSS, AMB, BLUE, render
TMP = os.environ.get("TMPDIR", "/tmp")
def chip(t, ratio):
    w = max(7, 4.2 * len(t.strip())); gap = 7 if t[:1].isspace() else 2
    return f'<span class="c" style="width:{w:.0f}px;margin-left:{gap}px;background:{colors.to_hex(pct_colour(ratio))}"></span>'
LINEF = "Focus your attention on paragraph [P{k}]."
allr = []; rows = ""
for bi, (c, side, cid) in enumerate(picks):
    r = byid[cid]; col = PURPLE if E[c][cid] > 0 else RED
    for k in range(1, 5): assert LINEF.format(k=k) in r["vers"][f"{c}{k}"]["user"]
    bub = ""
    for k in range(1, 5):
        pls = ""
        for m in range(4):
            a = np.array(r["vers"][f"{c}{k}"]["tok_L28"][m]); b = np.array(r["vers"][f"mention{k}"]["tok_L28"][m]); allr.extend((a / b).tolist())
            chips = "".join(chip(t, v) for t, v in zip(r["tokens"][m], a / b))
            on = k == m + 1
            pls += f'<div class="pr"><b class="lb{" k" if on else ""}">[P{m + 1}]</b><div class="pp{" on" if on else ""}">{chips}</div></div>'
        bub += f'<div class="ub g">{pls}<div class="var" style="color:{col}">{LINEF.format(k=k)}</div></div>'
    rows += (f'<div class="rh"><span class="big" style="color:{col}">{pct(E[c][cid])}</span><span class="rt">attention on the named paragraph, one typical question</span></div>'
             f'<div class="row">{bub}</div>')
allr = np.array(allr)
print(f"per-token ratio: {np.mean(np.abs(np.log2(allr)) > 2) * 100:.1f}% of chips beyond the colour range (x1/4 .. x4); 1st/99th pct {np.percentile(allr, 1):.2f} {np.percentile(allr, 99):.2f}")
stops = ",".join(f"{colors.to_hex(pct_colour(2.0 ** (-2 + 4 * i / 40)))} {i * 2.5}%" for i in range(41))
W = 2300; BW = 1500
ticks = "".join(f'<span class="tk" style="left:{BW * (x + 2) / 4:.0f}px">{pct(100 * (2.0 ** x - 1)) if x else "0%"}</span>' for x in (-2, -1, 0, 1, 2))
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:{W}px}}
.rh{{display:flex;align-items:baseline;gap:26px;margin:6px 0 10px 4px}} .rh.sec{{margin-top:34px}}
.big{{font-size:96px;font-weight:800;letter-spacing:-1px;line-height:1}} .rt{{font-size:50px;font-weight:700}}
.row{{display:flex;gap:28px}}
.ub.g{{flex:1;min-width:0;padding:18px 18px;gap:12px;font-size:46px}}
.pr{{display:flex;align-items:flex-start;gap:10px}}
.lb{{font-size:44px;line-height:1;padding:4px 4px;flex:0 0 auto;min-width:86px}}
.lb.k{{background:{AMB};color:{INK};border-radius:10px}}
.pp{{flex:1;background:#fff;border-radius:12px;padding:8px 8px 8px 4px;display:flex;flex-wrap:wrap;row-gap:4px;border:5px solid #fff}}
.pp.on{{border-color:{AMB}}}
.c{{display:inline-block;height:17px;border-radius:3px}}
.var{{font-size:44px;padding:8px 12px}}
.key{{margin:40px auto 0;width:{BW}px}}
.kt{{font-size:50px;font-weight:700;text-align:center;margin-bottom:14px;white-space:nowrap;margin-left:-300px;margin-right:-300px}}
.kb{{height:64px;border-radius:12px;background:linear-gradient(to right,{stops})}}
.ks{{position:relative;height:70px}} .tk{{position:absolute;top:10px;transform:translateX(-50%);font-size:50px;font-weight:800}}
</style></head><body>{rows.replace('<div class="rh">', '<div class="rh sec">', 2).replace('<div class="rh sec">', '<div class="rh">', 1)}
<div class="key"><div class="kt">change in attention on each token at layer 28, compared to only mentioning the paragraph</div><div class="kb"></div><div class="ks">{ticks}</div></div>
</body></html>"""
render(page, W, FIG, "grid_focus_one", TMP)
