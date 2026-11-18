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
    # ---------- violins on a labelled axis: zero = the line that only mentions the paragraph; mean with 95% bootstrap CI.
    # Broken y axis: one ignore prompt sits near +192%; the top strip shows it so no prompt is dropped.
    allz = np.concatenate([r["z"] for r in res]); BLO, BHI = -49, 96
    hiz = allz[allz > BHI]
    fig, (at, a) = plt.subplots(2, 1, figsize=(23, 10.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 7.5], hspace=0.04))
    fig.subplots_adjust(left=0.11, right=0.995, top=0.985, bottom=0.02)
    rng = np.random.default_rng(1); X = [0, 1.35]
    for x0, r in zip(X, res):
        z = r["z"]; colr = PURPLE if r["mean"] > 0 else RED
        bs = z[np.random.default_rng(0).integers(0, len(z), (20000, len(z)))].mean(1); lo95, hi95 = np.percentile(bs, [2.5, 97.5])
        r["ci"] = (lo95, hi95)
        vv = a.violinplot([z], positions=[x0], widths=0.45, showextrema=False)
        for bb in vv["bodies"]: bb.set_facecolor(colr); bb.set_alpha(0.18); bb.set_edgecolor(colr); bb.set_linewidth(2.5)
        xj = x0 + rng.uniform(-0.15, 0.15, len(z))
        for ax in (a, at): ax.scatter(xj, z, s=300, color=colr, alpha=0.5, edgecolors="none", zorder=3)
        a.plot([x0, x0], [lo95, hi95], color=INK, lw=16, solid_capstyle="butt", zorder=6)
        a.plot([x0 - 0.1, x0 + 0.1], [r["mean"]] * 2, color=INK, lw=7, zorder=7)
        eff = f"{r['mean']:+.0f}%".replace("-", "\u2212")
        pt = "p < 0.00001" if r["p"] * (r["nf"] + 1) < 1.5 else f"p = {r['p']:.1g}"
        assert r["p"] * (r["nf"] + 1) >= 1.5 or 1 / (r["nf"] + 1) < 1e-5
        up = r["mean"] > 0; ye = r["mean"] + 6 if up else -6   # effect, then p under the effect
        a.text(x0 + 0.27, ye, eff, fontsize=96, fontweight="bold", color=colr, ha="left", va="bottom" if up else "top")
        a.text(x0 + 0.29, ye - 1.5 if up else ye - 22, pt, fontsize=46, color=colr, ha="left", va="top")
    a.axhline(0, color=INK, lw=4, zorder=4)
    a.text(-0.98, -2, "only mentions [P%d]" % K, fontsize=46, color=INK, ha="left", va="top")
    a.set_xlim(-1.0, 2.15); a.set_xticks([])
    a.set_ylim(BLO, BHI); at.set_ylim(hiz.min() - 8, hiz.max() + 8)
    a.set_yticks([-40, -20, 0, 20, 40, 60, 80]); at.set_yticks([190])
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "\u2212") if v else "0")
    for ax in (a, at):
        ax.yaxis.set_major_formatter(fmt); ax.tick_params(axis="y", labelsize=44, width=4, length=14, colors=INK)
        ax.tick_params(axis="x", length=0)
        for s_ in ["top", "right", "bottom"]: ax.spines[s_].set_visible(False)
        ax.spines["left"].set_linewidth(4)
    for ax, y in ((at, 0), (a, 1)):   # break marks on the y axis
        ax.plot([-0.012, 0.012], [y - 0.006 * (8 if y == 0 else 1), y + 0.006 * (8 if y == 0 else 1)], transform=ax.transAxes, color=INK, lw=4, clip_on=False)
    a.set_ylabel("change in attention on [P%d]" % K, fontsize=48, labelpad=10)
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
for r in res: print(r["c"], "95% CI", np.round(r["ci"], 1))
