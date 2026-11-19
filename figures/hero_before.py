"""Hero figure, instruction placed once BEFORE the four paragraphs (before.py run on the dose contexts).
Adapted from hero_dose.py: hero figure from the dose run (Qwen3-32B, 150 arithmetic contexts).
Two columns: 'Focus your attention on paragraph [Pk].' and 'Completely ignore paragraph [Pk]. Do not look at it at all.'
Each violin: per-context % change in attention on the named paragraph (answer-token queries, band layers, mean over heads,
same measure as dose_analyze.py) vs a control: the mention-only line (hero.png) or no instruction (hero_none.png).
p: two-sided sign-flip test over contexts on the mean % change. Example bubble text: dose.json id 300 (real prompt)."""
import json, base64, subprocess, os, html, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK, pct_colour
D = "/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/qwen3-32b-before/"
meta = json.load(open(D + "meta.json")); L = meta["LAYERS"]
R = json.load(open(D + "dose.json")); by = {(r["id"], r["cond"]): r for r in R}
ids = sorted({r["id"] for r in R})
def pk(r): return np.array(r["att_mean"])[L].mean(0)[r["k"]]
M = {c: np.array([pk(by[i, c]) for i in ids]) for c in meta["conds"]}
for i in ids:
    assert len({by[i, c]["k"] for c in meta["conds"]}) == 1
RES = json.load(open(D + "before_results.json"))

def signflip(z, NF=200000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(z.mean()); hits = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(z))); hits += int((np.abs((s * z).mean(1)) >= obs).sum())
    return (hits + 1) / (NF + 1), NF

CONDS = [("b_focus", "focus your attention", PURPLE), ("b_ignore", "completely ignore", RED)]
EX = 300
ex = {c: by[EX, c] for c, _, _ in CONDS}
ANS = {c: ex[c]["ans_free"].strip() for c in ex}
K = ex["b_focus"]["k"] + 1
u = ex["b_focus"]["user"]; paras = [p.split("] ", 1)[1] for p in u.split("\n\n")[1:5]]
qline = u.split("\n\n")[5]; assert qline.startswith("Unrelated question")
LINE = {c: meta["conds"][c][1].format(k=K) for c, _, _ in CONDS}
for c in LINE: assert ex[c]["user"].startswith(LINE[c] + "\nHere are four paragraphs.")
for c in LINE: assert LINE[c] in ex[c]["user"] and qline in ex[c]["user"]

robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
AMB = "#F2B53A"; W = 2300
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
numbers = {}

from hero_cols import bubble_col, violins, key_svg, render, b64, CSS, labels_html
from pfmt import pstr
TMP = os.environ.get("TMPDIR", "/tmp")
for ctrl, ctag, out in [("b_mention", "mention", "hero_before")]:
    res = []
    for c, name, col in CONDS:
        z = 100 * (M[c] / M[ctrl] - 1)
        assert abs(z.mean() - RES[c[2:]]["before_vs_" + ("before_mention" if ctag == "mention" else "none")]["mean"]) < 1e-6
        p, nf = signflip(z)
        bs = z[np.random.default_rng(0).integers(0, len(z), (20000, len(z)))].mean(1)
        res.append(dict(c=c, z=np.log2(M[c] / M[ctrl]), mean=z.mean(), p=pstr(p, floor=1 / (nf + 1)), ci=np.percentile(bs, [2.5, 97.5])))
        numbers[(ctag, c)] = (round(z.mean(), 1), p, nf, round(100 * (z > 0).mean()))
    # ---------- violins: per-prompt % change vs the line that only mentions the paragraph (also placed before), log-ratio axis
    VW = W - 52; VH = 720; vp = os.path.join(TMP, f"_v{out}.png")
    info = violins(res, VW, VH, vp, skip=(18, 250), mean_mark=True, tick_right=190, cxs=[930, 3 * VW / 4 + 9])   # focus violin right of the baseline row (0% = and its box on one line)
    base = f'<span>0% =</span><div class="cub"><div class="var">{html.escape(meta["conds"][ctrl][1].format(k=K))}</div></div>'
    cols = "".join(bubble_col(name, col, paras, K, LINE[c], qline, ANS[c], line_first=True,
                              ans_note=f"attention to [P{K}] is measured<br>from these answer tokens") for c, name, col in CONDS)
    HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>{CSS} body{{width:{W}px}} .bl.wide .cub{{max-width:660px}} .bl.wide .var{{text-wrap:balance}} .vio{{width:{VW}px}}</style></head><body>
<div class="cols">{cols}</div>
<div class="vw"><img class="vio" src="data:image/png;base64,{b64(vp)}">
<div class="ov bl wide" style="left:8px;top:{info["y0"] + 18:.0f}px">{base}</div>
<div class="ov" style="left:{info["cx"][1] + 60:.0f}px;top:{info["Y"](np.log2(1.75)):.0f}px">{key_svg()}</div>{labels_html(info, res)}</div></body></html>'''
    render(HTML, W, HERE, out, TMP); os.remove(vp)
for k, v in numbers.items(): print(k, "mean %+.1f%%  p_signflip=%.2g (of %d flips)  share>0 %d%%" % v)
for r in res: print(r["c"], "95% CI", np.round(r["ci"], 1))
