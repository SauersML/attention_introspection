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
from pal import PURPLE, RED, INK, DIV, norm as dnorm
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

def short(p, n=3): return " ".join(p.split()[:n]) + " …"
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
AMB = "#F2B53A"; W = 2300
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
numbers = {}

for ctrl, ctag, out in [("b_mention", "mention", "hero_before")]:
    res = []
    for c, name, col in CONDS:
        z = 100 * (M[c] / M[ctrl] - 1)
        assert abs(z.mean() - RES[c[2:]]["before_vs_" + ("before_mention" if ctag == "mention" else "none")]["mean"]) < 1e-6
        p, nf = signflip(z)
        res.append(dict(c=c, z=z, mean=z.mean(), p=p, nf=nf, above=float((z > 0).mean())))
        numbers[(ctag, c)] = (round(z.mean(), 1), p, nf, round(100 * (z > 0).mean()))
    # ---------- violins
    fig, a = plt.subplots(figsize=(23, 8.2)); fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    allz = np.concatenate([r["z"] for r in res]); DN = dnorm(np.percentile(allz, 1), np.percentile(np.abs(allz), 97))
    rng = np.random.default_rng(1); X = [0, 1.35]; texts = []
    for x0, r in zip(X, res):
        z = r["z"]; up = r["mean"] > 0; colr = PURPLE if up else RED
        vv = a.violinplot([z], positions=[x0], widths=0.7, showextrema=False)
        for bb in vv["bodies"]: bb.set_facecolor(colr); bb.set_alpha(0.22); bb.set_edgecolor(colr); bb.set_linewidth(2.5)
        xj = x0 + rng.uniform(-0.25, 0.25, len(z))
        a.scatter(xj, z, s=650, c=DIV(DN(z)), alpha=0.45, edgecolors="none", zorder=3); a.scatter(xj, z, s=6, color="black", zorder=4)
        eff = f"{r['mean']:+.0f}% attention".replace("-", "\u2212")
        pt = "p < 0.00001" if r["p"] * (r["nf"] + 1) < 1.5 else f"p = {r['p']:.1g}"
        assert r["p"] * (r["nf"] + 1) >= 1.5 or 1 / (r["nf"] + 1) < 1e-5
        r["eff"], r["pt"] = eff, pt
        if up:   # effect above the dots, p under the effect
            texts.append(a.annotate(pt, (x0, z.max()), xytext=(0, 30), textcoords="offset points", fontsize=44, color=colr, ha="center", va="bottom"))
            texts.append(a.annotate(eff, (x0, z.max()), xytext=(0, 90), textcoords="offset points", fontsize=100, fontweight="bold", color=colr, ha="center", va="bottom"))
        else:    # effect below the dots, p under the effect
            texts.append(a.annotate(eff, (x0, z.min()), xytext=(0, -26), textcoords="offset points", fontsize=100, fontweight="bold", color=colr, ha="center", va="top"))
            texts.append(a.annotate(pt, (x0, z.min()), xytext=(0, -138), textcoords="offset points", fontsize=44, color=colr, ha="center", va="top"))
    a.axhline(0, color=INK, lw=4, zorder=4)
    a.set_xlim(-0.66, 2.01); a.set_xticks([]); a.set_yticks([])
    for s_ in a.spines.values(): s_.set_visible(False)
    lo, hi = min(allz.min(), 0), max(allz.max(), 0); a.set_ylim(lo, hi)
    for _ in range(6):   # grow y-limits until every label fits
        fig.canvas.draw(); rd = fig.canvas.get_renderer(); inv = a.transData.inverted()
        ys = [inv.transform(t.get_window_extent(rd).get_points())[:, 1] for t in texts]
        ylo = min(lo, min(y.min() for y in ys)); yhi = max(hi, max(y.max() for y in ys))
        pad = 0.01 * (yhi - ylo); a.set_ylim(ylo - pad, yhi + pad)
    fig.savefig(os.path.join(HERE, f"v{out}.png"), dpi=110, facecolor="white"); plt.close(fig)
    # ---------- chat columns + violins
    VIO = base64.b64encode(open(os.path.join(HERE, f"v{out}.png"), "rb").read()).decode()
    cols = []
    for c, name, col in CONDS:
        pl = "".join(f'<div class="pl"><b class="{"k" if i + 1 == K else ""}">[P{i+1}]</b> {html.escape(short(p, 2 if i == 0 else 3))}</div>' for i, p in enumerate(paras))
        cols.append(f'''<div class="col"><div class="cname" style="color:{col}">{name}</div>
  <div class="ub"><div class="var" style="color:{col}">{html.escape(LINE[c])}</div><div>Here are four paragraphs.</div>{pl}
    <div>{html.escape(qline).replace(" + ", "&nbsp;+&nbsp;")}</div></div>
  <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{ANS[c]}</div></div></div>''')
    HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.cols{{display:flex;gap:36px}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:76px;font-weight:800;margin:0 0 12px 4px;white-space:nowrap}}
.ub{{flex:1;background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap;overflow:hidden}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:12px}}
.rob{{width:110px;height:110px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:70px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.vio{{width:100%;display:block;margin-top:10px}}
</style></head><body><div class="cols">{"".join(cols)}</div><img class="vio" src="data:image/png;base64,{VIO}"></body></html>'''
    hp = os.path.join(HERE, f"{out}.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, f"{out}_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3200",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, out + ".png")); im.save(os.path.join(HERE, out + ".jpg"), quality=92); print(out, im.size)
for k, v in numbers.items(): print(k, "mean %+.1f%%  p_signflip=%.2g (of %d flips)  share>0 %d%%" % v)
