"""Figure for results/monitor/LOGPROB.md (Qwen3-32B, 300 prompts with four paragraphs).
Top: one real prompt (id 173): the chat, the self-report question, the report's probabilities for P1..P4 (after the
prefilled "P") beside the true attention (the preregistered primary truth: attention from answer tokens to each paragraph,
per token of the paragraph, averaged over heads and layers 21-53, as percent of the four values' total). Bottom left: per-prompt Spearman between report score and true attention, asked
right after answering, self vs the same model reading the transcript; black bar = 95% bootstrap CI of the mean.
Bottom right: held-out gain in within-prompt rank correlation from adding the self score to a model that already has the
reader's scores for every wording asked at that timing, plus position and length (logprob.py "gain_all"), dot + 95% bootstrap CI over prompts.
usage: monitor_logprob_fig.py results/monitor/logprob_results.json SHARD_DIR [SHARD_DIR ...]   (writes figures/monitor_logprob.png)"""
import json, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, INK
ORANGE = "#E07B24"
J = json.load(open(sys.argv[1])); DIRS = sys.argv[2:]
TR = "norm"  # preregistered primary truth
RES, PP, IDS = J["res"][TR], J["perprompt"][TR], J["ids"]
C, H, G = {}, {}, {}
for d in DIRS:
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    H.update(json.load(open(os.path.join(d, "hf1.json")))); G.update(json.load(open(os.path.join(d, "gen1.json"))))
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
AMB = "#F2B53A"; W = 2400; PAD = 30
px = lambda v: v * 72 / 100

# ---------------- example prompt ----------------
EX = "173"; c = C[EX]
truth = np.array(H[EX]["truth"][TR])[21:54].mean(0); truth = truth / truth.sum()
rep = np.array(H[EX]["rep"]["after|most|self"])
r_ex = PP["after|most"]["self"][IDS.index(EX)]
print("example", EX, "report", np.round(rep, 3), "truth", np.round(truth, 3), "rank corr", round(r_ex, 3))
def short(p, n): return " ".join(p.split()[:n]) + " …"
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
QTEXT = "Which paragraph did you attend to most while writing your answer?"
def bars(p, col, pct=True):
    w, h, bw, top = 860, 340, 150, 260; o = []
    for k in range(4):
        x = 40 + k * 205; hh = 200 * p[k] / max(p.max(), 1e-9)
        o.append(f'<rect x="{x}" y="{top - hh:.1f}" width="{bw}" height="{hh:.1f}" rx="6" fill="{col}"/>')
        o.append(f'<text x="{x + bw / 2}" y="{top + 56}" font-size="46" font-weight="700" text-anchor="middle" fill="{INK}">P{k + 1}</text>')
        o.append(f'<text x="{x + bw / 2}" y="{top - hh - 12:.1f}" font-size="42" text-anchor="middle" fill="{INK}">{100 * p[k]:.0f}%</text>')
    o.append(f'<line x1="10" x2="{w - 10}" y1="{top}" y2="{top}" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" viewBox="0 -10 {w} {h}">{"".join(o)}</svg>'
paras = "".join(f'<div class="pl"><b>[P{k + 1}]</b> {html.escape(short(p, 3))}</div>' for k, p in enumerate(c["paras"]))
TOP = f'''<div class="top">
<div class="chat"><div class="mname">Qwen3-32B, 300 prompts</div>
 <div class="ub">Here are four paragraphs.{paras}<div class="q">{html.escape(c["task"])}</div></div>
 <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(G[EX]["ans_text"])}</div></div>
 <div class="ub ub2">{QTEXT} …</div>
 <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb mp">P…</div></div>
</div>
<div class="right">
 <div class="box"><div class="bt" style="color:{PURPLE}">report probability</div>{bars(rep, PURPLE)}</div>
 <div class="box"><div class="bt">true attention</div>{bars(truth, AMB)}</div>
</div></div>'''

# ---------------- bottom panels ----------------
ROWS = [("after|most", "right after"), ("before|most", "before answering"), ("later|most", "one turn later"), ("think|most", "with thinking"),
        ("fb4|most", "after 4 true examples"), ("fb16|most", "after 16 true examples"),
        ("after|rate", "right after"), ("before|rate", "before answering"), ("later|rate", "one turn later"),
        ("fb4|rate", "after 4 true examples"), ("fb16|rate", "after 16 true examples"),
        ("after|least", "least attended"), ("after|free", "free answer")]
ROWS = [r for r in ROWS if r[0] in RES]
GROUP = {0: "which paragraph most?", [k for k, _ in ROWS].index("after|rate"): "0–10 rating", [k for k, _ in ROWS].index("after|least"): "other questions"}
BH = 1560
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK})
fig = plt.figure(figsize=(W / 100, BH / 100), dpi=100)
F = lambda x, y: (x / W, y / BH)
# left: violins
L0, LW, B0, BHh = 190, 720, 300, 1160
ax = fig.add_axes([L0 / W, B0 / BH, LW / W, BHh / BH])
rng = np.random.default_rng(1)
for j, (who, col, lab) in enumerate([("self", PURPLE, "self-report"), ("reader", ORANGE, "same model\nreading the\ntranscript")]):
    z = np.array(PP["after|most"][who]); st = RES["after|most"][who]
    kde = gaussian_kde(z, bw_method=0.3); yy = np.linspace(-1.05, 1.05, 300); dd = kde(yy); dd = dd / dd.max() * 0.42
    ax.fill_betweenx(yy, j - dd, j + dd, color=col, alpha=0.12, lw=0); ax.plot(np.r_[j - dd, (j + dd)[::-1]], np.r_[yy, yy[::-1]], color=col, lw=2.5, alpha=0.6)
    xj = j + rng.uniform(-1, 1, len(z)) * kde(z) / kde(yy).max() * 0.40; yj = z + rng.uniform(-0.035, 0.035, len(z))
    ax.scatter(xj, yj, s=120, color=col, alpha=0.45, edgecolors="none", zorder=3); ax.scatter(xj, yj, s=4, color="black", zorder=4)
    ax.plot([j, j], [st["lo"], st["hi"]], color=INK, lw=10, solid_capstyle="butt", zorder=6)
    ax.plot([j - 0.17, j + 0.17], [st["mean"]] * 2, color=INK, lw=6, zorder=6)
    ax.text(j + 0.2, st["mean"], f"{st['mean']:.2f}", ha="left", va="center", fontsize=px(54), fontweight="bold", color=col, zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax.text(j, -1.16, lab, ha="center", va="top", fontsize=px(48), fontweight="bold", color=col, linespacing=1.05)
ax.set_xlim(-0.55, 1.55); ax.set_ylim(-1.1, 1.1); ax.set_xticks([])
ax.set_yticks([-1, 0, 1]); ax.set_yticklabels(["\u22121", "0", "1"], fontsize=px(46)); ax.tick_params(length=0, pad=10)
for s_ in ["top", "right", "bottom"]: ax.spines[s_].set_visible(False)
ax.spines["left"].set_linewidth(3); ax.axhline(0, color=INK, lw=2, zorder=0)
fig.text(*F(50, B0 + BHh / 2), "rank correlation with true attention", rotation=90, ha="center", va="center", fontsize=px(48))
fig.text(*F(L0 + LW / 2, BH - 14), "asked right after answering", ha="center", va="top", fontsize=px(50), fontweight="bold")
# right: gains
R0, RW = 1560, 760
ax2 = fig.add_axes([R0 / W, B0 / BH, RW / W, BHh / BH])
ys = []; y = 0
for n, (k, _) in enumerate(ROWS):
    if n in GROUP: y += 1
    ys.append(y); y += 1
YMAX = y
lab_tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax2.transData)
for n, (k, lab) in enumerate(ROWS):
    q = RES[k]; yv = YMAX - ys[n]
    ax2.plot([q["gain_all"]["lo"], q["gain_all"]["hi"]], [yv, yv], color=PURPLE, lw=7, solid_capstyle="round", zorder=3)
    ax2.scatter([q["gain_all"]["mean"]], [yv], s=520, color=PURPLE, zorder=4, edgecolors="white", linewidths=2)
    ax2.text((R0 - 24) / W, yv, lab, transform=lab_tr, ha="right", va="center", fontsize=px(46))
for n, g in GROUP.items():
    ax2.text((R0 - 24) / W, YMAX - ys[n] + 1, g, transform=lab_tr, ha="right", va="center", fontsize=px(48), fontweight="bold")
ax2.axvline(0, color=INK, lw=3, zorder=2)
ax2.set_ylim(0.45, YMAX + 0.6); ax2.set_yticks([])
lo = min(RES[k]["gain_all"]["lo"] for k, _ in ROWS); hi = max(RES[k]["gain_all"]["hi"] for k, _ in ROWS)
ax2.set_xlim(min(lo, -0.02) - 0.004, hi + 0.004)
ax2.set_xticks([-0.02, 0, 0.02]); ax2.set_xticklabels(["\u22120.02", "0", "+0.02"], fontsize=px(46)); ax2.tick_params(length=8, width=3, pad=8)
for s_ in ["top", "right", "left"]: ax2.spines[s_].set_visible(False)
ax2.spines["bottom"].set_linewidth(3)
fig.text(*F(R0 + RW / 2, 18), "rank correlation the self-report adds over\nthe same model reading the transcript,\nasked every way at the same time", ha="center", va="bottom", fontsize=px(46), linespacing=1.1)
VP = os.path.join(HERE, "monitor_logprob_bottom.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
VIO = base64.b64encode(open(VP, "rb").read()).decode()
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:24px {PAD}px 0}}
.top{{display:flex;gap:40px;align-items:center}}
.chat{{width:1330px}}
.ub{{background:#1F8BFF;color:#fff;border-radius:30px;padding:16px 24px;font-size:44px;line-height:1.2;margin-left:100px}}
.ub2{{margin-top:14px}} .pl{{white-space:nowrap}} .q{{margin-top:6px}}
.mrow{{display:flex;align-items:center;gap:14px;margin-top:12px}}
.rob{{width:86px;height:86px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:6px 28px;font-size:56px;font-weight:700}}
.mname{{font-size:50px;font-weight:800;margin:0 0 10px 100px}} .mp{{border-color:{PURPLE};color:{PURPLE}}}
.right{{display:flex;flex-direction:column;gap:26px}}
.box{{border:4px solid {INK};border-radius:28px;padding:14px 22px 0}}
.bt{{font-size:50px;font-weight:800}}
.bot{{width:{W}px;display:block;margin:30px 0 0 -{PAD}px}}
</style></head><body>{TOP}<img class="bot" src="data:image/png;base64,{VIO}"></body></html>'''
hp = os.path.join(HERE, "monitor_logprob.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "monitor_logprob_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3200",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 24)); im.save(os.path.join(HERE, "monitor_logprob.png")); print("monitor_logprob.png", im.size)
os.remove(raw); os.remove(hp); os.remove(VP)
