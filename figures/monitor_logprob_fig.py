"""Figure for results/monitor_inorder/RESULT.md and results/monitor_shuffled/LOGPROB.md (Qwen3-32B, 300 prompts with four paragraphs).
Top: one real prompt (id 173), drawn as the two setups: left, the self-report (the same chat, asked right after answering); right, the
same model reading the transcript (the reader prompt's real opening line, the conversation quoted, the reader's question). Each reply
shows that setup's probabilities for P1..P4 (after the prefilled "P"); under them the true attention (the preregistered primary truth:
attention from answer tokens to each paragraph, per token of the paragraph, averaged over heads and layers 21-53, as percent of the
four values' total). Bottom left: per-prompt Spearman between report score and true attention, asked right after answering, self vs
reader, one violin under each setup; big number = mean; p = paired two-sided sign-flip test of self vs reader (200,000 flips).
(The held-out gain over every wording, logprob.py "gain_all", is reported in LOGPROB.md, not drawn.)
Third violin: a model of another family reading the same transcript (monitor/reader_xmodel.py, results/monitor_inorder/reader_xmodel_gemma4_scored.json).
usage: monitor_logprob_fig.py results/monitor_inorder/logprob_results.json OUT.png results/monitor_inorder/reader_xmodel_gemma4_scored.json SHARD_DIR [SHARD_DIR ...]"""
import json, os, sys, html, base64, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, INK
from pfmt import pstr
BLUE = "#2F7FC1"   # the reader: the blue end of the shared palette, darkened enough for text
GREEN = "#1A9E77"  # the reader of another model family
J = json.load(open(sys.argv[1])); OUTP = os.path.abspath(sys.argv[2]); X = None if sys.argv[3] == "-" else json.load(open(sys.argv[3])); DIRS = sys.argv[4:]
TR = "norm"  # preregistered primary truth
RES, PP, IDS = J["res"][TR], J["perprompt"][TR], J["ids"]
C, H, G = {}, {}, {}
for d in DIRS:
    for c in json.load(open(os.path.join(d, "contexts.json"))): C[str(c["id"])] = c
    H.update(json.load(open(os.path.join(d, "hf1.json")))); G.update(json.load(open(os.path.join(d, "gen1.json"))))
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
AMB = "#F2B53A"; W = 2300; PAD = 26
px = lambda v: v * 72 / 100

# ---------------- example prompt ----------------
EX = "173"; c = C[EX]; iex = IDS.index(EX)
truth = np.array(H[EX]["truth"][TR])[21:54].mean(0); truth = truth / truth.sum()
rep = {"self": np.array(H[EX]["rep"]["after|most|self"]), "reader": np.array(H[EX]["rep"]["after|most|reader"])}
r_ex = {w: PP["after|most"][w][iex] for w in rep}
print("example", EX, {w: np.round(v, 3) for w, v in rep.items()}, "truth", np.round(truth, 3), "rank corr", {w: round(v, 3) for w, v in r_ex.items()})
def short(p, n): return " ".join(p.split()[:n]) + " \u2026"
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
QS = "Which paragraph did you attend to most while writing your answer?"
QR = "Which paragraph did the assistant attend to most while writing its answer?"
RINTRO = "Below is a conversation between a user and an AI assistant (Qwen3-32B)."
def bars(p, col, w=600, h=250):
    """four bars P1..P4, height = share, % above"""
    bw = 104; gap = (w - 4 * bw) / 4; top = h - 64; o = []
    for k in range(4):
        x = gap / 2 + k * (bw + gap); hh = (top - 62) * p[k] / max(p.max(), 1e-9)
        if hh > 0.5: o.append(f'<rect x="{x:.1f}" y="{top - hh:.1f}" width="{bw}" height="{hh:.1f}" rx="8" fill="{col}"/>')
        o.append(f'<text x="{x + bw / 2:.1f}" y="{top + 52}" font-size="46" font-weight="800" text-anchor="middle" fill="{INK}">P{k + 1}</text>')
        o.append(f'<text x="{x + bw / 2:.1f}" y="{top - hh - 12:.1f}" font-size="46" font-weight="700" text-anchor="middle" fill="{INK}">{100 * p[k]:.0f}%</text>')
    o.append(f'<line x1="4" x2="{w - 4}" y1="{top}" y2="{top}" stroke="{INK}" stroke-width="3"/>')
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif">{"".join(o)}</svg>'
paras = "".join(f'<div class="pl"><b>[P{k + 1}]</b> {html.escape(short(p, 3))}</div>' for k, p in enumerate(c["paras"]))
TASK = html.escape(c["task"]).replace("[P4]", "<b>[P4]</b>").replace("[P2]", "<b>[P2]</b>")
ANS = html.escape(G[EX]["ans_text"])
LEFT = f'''<div class="col"><div class="cname" style="color:{PURPLE}">self-report</div>
 <div class="ub">Here are four paragraphs.{paras}<div class="tk">{TASK}</div></div>
 <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{ANS}</div></div>
 <div class="ub q"><div class="var" style="color:{PURPLE}">{QS} \u2026</div></div>
 <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb rp" style="border-color:{PURPLE}">{bars(rep["self"], PURPLE)}</div></div></div>'''
tparas = "".join(f'<div class="pl"><b>[P{k + 1}]</b> {html.escape(short(p, 3))}</div>' for k, p in enumerate(c["paras"]))
RIGHT = f'''<div class="col"><div class="cname" style="color:{BLUE}">same model reading the transcript</div>
 <div class="ub">{RINTRO}
  <div class="tr"><div><b>User:</b> Here are four paragraphs.</div>{tparas}<div class="tk">{TASK}</div><div class="ta"><b>Assistant:</b> {ANS}</div></div>
  <div class="var" style="color:{BLUE}">{QR} \u2026</div></div>
 <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb rp" style="border-color:{BLUE}">{bars(rep["reader"], BLUE)}</div></div></div>'''

# ---------------- bottom panels (matplotlib, px coordinates at dpi 100) ----------------
ROWS = [("after|most", "right after"), ("before|most", "before answering"), ("later|most", "one turn later"), ("think|most", "with thinking"),
        ("fb4|most", "after 4 true examples"), ("fb16|most", "after 16 true examples"),
        ("after|rate", "right after"), ("before|rate", "before answering"), ("later|rate", "one turn later"),
        ("fb4|rate", "after 4 true examples"), ("fb16|rate", "after 16 true examples"),
        ("after|least", "least attended"), ("after|free", "free answer")]
ROWS = [r for r in ROWS if r[0] in RES]
GROUP = {0: "which paragraph most?", [k for k, _ in ROWS].index("after|rate"): "0\u201310 rating", [k for k, _ in ROWS].index("after|least"): "other questions"}
BW = W - 2 * PAD; BH = 1560 - (0 if X else 80)
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(BW / 100, BH / 100), dpi=100)
# left: violins of per-prompt rank correlation
SH = 0 if X else 80                   # no second bracket row without the third violin
AXL, B0, T0 = 300, 40, 640 - SH           # px: axes left edge (room for the y label and ticks), bottom margin, top band
AXW = BW - AXL - 10
Y_NAME, Y_BR2, Y_BR1, Y_NUM = 80, 280, 360 - SH, 410 - SH     # px from the top: names, the two p brackets, big numbers
ax = fig.add_axes([AXL / BW, B0 / BH, AXW / BW, (BH - B0 - T0) / BH])
kx = fig.add_axes([0, 0, 1, 1], facecolor="none", zorder=-1); kx.set_xlim(0, BW); kx.set_ylim(BH, 0); kx.axis("off")
rng = np.random.default_rng(1)
dsr = np.subtract(PP["after|most"]["self"], PP["after|most"]["reader"]); obs = abs(dsr.mean())
NF = 200000
PV = (sum(int((np.abs((np.random.default_rng(k).choice([-1.0, 1.0], (20000, len(dsr))) * dsr).mean(1)) >= obs - 1e-12).sum()) for k in range(10)) + 1) / (NF + 1)
PTXT = pstr(PV, floor=1 / (NF + 1))
print("self - reader", round(dsr.mean(), 4), "p", PV, PTXT)
def flip(x):
    x = np.asarray(x, float); o = abs(x.mean())
    return (sum(int((np.abs((np.random.default_rng(k).choice([-1.0, 1.0], (20000, len(x))) * x).mean(1)) >= o - 1e-12).sum()) for k in range(10)) + 1) / (NF + 1)
PV2 = flip(np.subtract(PP["after|most"]["self"], X["perprompt"]["x_reader"])) if X else None; PTXT2 = pstr(PV2, floor=1 / (NF + 1)) if X else None
P0 = flip(PP["after|most"]["self"]); P0TXT = pstr(P0, floor=1 / (NF + 1))
print("self - x reader p", PV2, PTXT2, "self vs 0 p", P0, P0TXT)
# three violins: the self-report, Qwen3-32B reading the transcript, and a model of another family reading the same transcript
ARMS = [("self", PURPLE, "Qwen3-32B\nself-report", PP["after|most"]["self"]), ("reader", BLUE, "Qwen3-32B\nreading the text", PP["after|most"]["reader"]),
        ]
if X:
    assert np.allclose(X["perprompt"]["self"], PP["after|most"]["self"]) and np.allclose(X["perprompt"]["qwen_reader"], PP["after|most"]["reader"])
    ARMS.append(("x", GREEN, {"google/gemma-4-31B-it": "Gemma 4 31B"}.get(X["model"], X["model"].split("/")[-1]) + "\nreading the text", X["perprompt"]["x_reader"]))
XC = [0.5 + j for j in range(len(ARMS))]; XL = (0.0, float(len(ARMS)))
X2P = lambda x: AXL + AXW * (x - XL[0]) / (XL[1] - XL[0])          # data x -> page px
for j, (who, col, name, z) in enumerate(ARMS):
    z = np.array(z); x0 = XC[j]
    kde = gaussian_kde(z, bw_method=0.3); yy = np.linspace(-1.0, 1.0, 300); dd = kde(yy); dd = dd / dd.max() * 0.40
    ax.fill_betweenx(yy, x0 - dd, x0 + dd, color=col, alpha=0.18, lw=0); ax.plot(np.r_[x0 - dd, (x0 + dd)[::-1], x0 - dd[:1]], np.r_[yy, yy[::-1], yy[:1]], color=col, lw=2.5)
    xj = x0 + rng.uniform(-1, 1, len(z)) * kde(z) / kde(yy).max() * 0.36; yj = np.clip(z + rng.uniform(-0.03, 0.03, len(z)), -1, 1)
    ax.scatter(xj, yj, s=330, color=col, alpha=0.40, edgecolors="none", zorder=3); ax.scatter(xj, yj, s=5, color="black", zorder=4)
    kx.text(X2P(x0), Y_NAME, name, ha="center", va="top", fontsize=px(54), fontweight="bold", color=col, linespacing=1.05)
    kx.text(X2P(x0), Y_NUM, f"{z.mean():.2f}", ha="center", va="top", fontsize=px(100), fontweight="bold", color=col)
# chance: rank correlation 0
ax.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=6)
ax.text(XL[1], 0.025, "chance", ha="right", va="bottom", fontsize=px(56), fontweight="bold", color=INK, zorder=7,
        bbox=dict(fc="white", ec="none", pad=2, alpha=0.85))
kx.text(X2P(XC[0]), Y_NUM + 118, "above chance,\n" + P0TXT, ha="center", va="top", fontsize=px(50), color=PURPLE, fontweight="bold", linespacing=1.05)
ax.set_xlim(*XL); ax.set_ylim(-1.06, 1.06); ax.set_xticks([])
ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"], fontsize=px(50), fontweight="bold", color=INK)
ax.tick_params(axis="y", length=14, width=3, color=INK, pad=8)
ax.set_ylabel("rank correlation with\ntrue attention, per prompt", fontsize=px(52), fontweight="bold", color=INK, labelpad=18, linespacing=1.1)
for k_, s_ in ax.spines.items(): s_.set_visible(k_ == "left")
ax.spines["left"].set_linewidth(3); ax.spines["left"].set_bounds(-1, 1)
# paired p brackets: self vs each reader
for j, yb, ptxt in [(1, Y_BR1, PTXT), (2, Y_BR2, PTXT2)][:len(ARMS) - 1]:
    xl, xr = X2P(XC[0]), X2P(XC[j])
    kx.plot([xl, xl, xr, xr], [yb + 26, yb, yb, yb + 26], color=INK, lw=4)
    kx.text((xl + xr) / 2 + (0 if j == 1 else (xr - xl) / 4), yb - 6, ptxt, ha="center", va="bottom", fontsize=px(52), color=INK, bbox=dict(fc="white", ec="none", pad=3))
# key for the dots, top left of the violin panel, left of the first name
KY = Y_NAME + 34; kx.scatter([40], [KY], s=330, color=PURPLE, alpha=0.75, edgecolors="none"); kx.scatter([40], [KY], s=5, color="black")
kx.text(70, KY, "= one prompt", ha="left", va="center", fontsize=px(52), fontweight="bold", color=INK)
kx.text(BW / 2, 6, "which paragraph did the model attend to most? asked right after answering", ha="center", va="top", fontsize=px(52), fontweight="bold", color=INK)
VP = os.path.join(HERE, "monitor_logprob_bottom.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
VIO = base64.b64encode(open(VP, "rb").read()).decode()
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px {PAD}px 10px}}
.cols{{display:flex;gap:36px;align-items:stretch}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:66px;font-weight:800;margin:0 0 12px 4px;white-space:nowrap}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:20px 26px;font-size:46px;line-height:1.18;display:flex;flex-direction:column;gap:4px}}
.ub.q{{margin-top:16px;padding:14px}}
.pl{{white-space:nowrap;overflow:hidden}} .tk{{margin-top:4px}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18;margin-top:6px}}
.tr{{border:3px solid rgba(255,255,255,.85);border-radius:20px;padding:12px 18px;margin:8px 0 2px;display:flex;flex-direction:column;gap:4px}}
.ta{{margin-top:6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:14px}}
.col .mrow:last-child{{margin-top:auto;padding-top:16px}}
.rob{{width:110px;height:110px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:8px 30px;font-size:58px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.mb.rp{{border-width:5px;padding:6px 14px 0}}
.truth{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:40px;margin-top:26px}}
.tl{{font-size:56px;font-weight:800;text-align:right;line-height:1.1}}
.tbox{{border:4px solid {INK};border-radius:28px;padding:6px 14px 0}}
.key{{display:flex;align-items:center;gap:14px;font-size:52px;font-weight:800;white-space:nowrap}}
.dot{{display:inline-block;width:44px;height:44px;border-radius:50%;background:{PURPLE};opacity:.75;position:relative}}
.dot::after{{content:"";position:absolute;left:19px;top:19px;width:6px;height:6px;border-radius:50%;background:#000}}
.ring{{display:inline-block;width:46px;height:46px;border-radius:50%;border:5px solid {INK}}}
.bot{{width:{W - 2 * PAD}px;display:block;margin-top:30px}}
</style></head><body><div class="cols">{LEFT}{RIGHT}</div>
<div class="truth"><div class="tl">true attention,<br>measured inside<br>the model</div><div class="tbox">{bars(truth, AMB)}</div><div></div></div>
<img class="bot" src="data:image/png;base64,{VIO}"></body></html>'''
hp = os.path.join(HERE, "monitor_logprob.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "monitor_logprob_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3600",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 24)); im.save(OUTP); print(OUTP, im.size, "aspect", round(im.size[1] / im.size[0], 2))
os.remove(raw); os.remove(hp); os.remove(VP)
