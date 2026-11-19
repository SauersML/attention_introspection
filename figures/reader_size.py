"""Does a smaller reader read the transcript worse? Qwen3-32B wrote 300 transcripts (four paragraphs, in order); two other models read each
transcript and are asked which paragraph the assistant attended to most (monitor/reader_xmodel.py). Per prompt: rank correlation between
the reader's probabilities for P1..P4 and Qwen3-32B's true attention (preregistered primary truth). Violins: Gemma 4 31B vs Qwen3-8B
reading the same transcripts; big number = mean; p = paired two-sided sign-flip test of the per-prompt difference (200,000 flips).
usage: reader_size.py results/monitor_inorder/reader_xmodel_gemma4_scored.json results/monitor_inorder/reader_xmodel_qwen3_8b_scored.json figures/reader_size.png"""
import json, os, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK
from pfmt import pstr
GREEN, ORANGE, PURPLE = "#1A9E77", "#E07B24", "#5E2B97"
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); OUT = sys.argv[3]
assert A["ids"] == B["ids"] and A["model"] == "google/gemma-4-31B-it" and B["model"] == "Qwen/Qwen3-8B"
ga, qb = np.array(A["perprompt"]["x_reader"], float), np.array(B["perprompt"]["x_reader"], float)
d = ga - qb; NF = 200000
PV = (sum(int((np.abs((np.random.default_rng(k).choice([-1.0, 1.0], (20000, len(d))) * d).mean(1)) >= abs(d.mean()) - 1e-12).sum()) for k in range(10)) + 1) / (NF + 1)
PT = pstr(PV, floor=1 / (NF + 1)); print("means", ga.mean(), qb.mean(), "diff", d.mean(), "p", PV, PT)

W, H = 2300, 1800; px = lambda v: v * 72 / 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
kx = fig.add_axes([0, 0, 1, 1], facecolor="none"); kx.set_xlim(0, W); kx.set_ylim(H, 0); kx.axis("off")
# method strip: three literal steps
FS = px(50)
def card(x, y, w, h, fc, ec, txt, col=INK, fw="bold"):
    kx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=30", fc=fc, ec=ec, lw=5))
    kx.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=FS, fontweight=fw, color=col, linespacing=1.15)
Y0, CH = 30, 250
card(30, Y0, 600, CH, "white", PURPLE, "the transcript\nQwen3-32B wrote", PURPLE)
card(760, Y0, 560, CH, "white", INK, "another model\nreads the\ntranscript")
card(1450, Y0, 820, CH, "#1F8BFF", "#1F8BFF", "\u201cWhich paragraph did\nthe assistant attend\nto most?\u201d", "white")
for x0, x1 in [(640, 750), (1330, 1440)]:
    kx.annotate("", xy=(x1, Y0 + CH / 2), xytext=(x0, Y0 + CH / 2), arrowprops=dict(arrowstyle="-|>,head_length=0.9,head_width=0.5", lw=6, color=INK))
# violins
AXL, AXT, AXB = 330, 750, 40
ax = fig.add_axes([AXL / W, AXB / H, (W - AXL - 30) / W, (H - AXT - AXB) / H])
XL = (0.0, 2.0); X2P = lambda x: AXL + (W - AXL - 30) * (x - XL[0]) / (XL[1] - XL[0])
rng = np.random.default_rng(1)
ARMS = [(GREEN, "Gemma 4 31B\nreading the text", ga), (ORANGE, "Qwen3-8B\nreading the text", qb)]
YN, YBR, YNUM = 320, 530, 560
for j, (col, name, z) in enumerate(ARMS):
    x0 = 0.5 + j; kde = gaussian_kde(z, bw_method=0.3); yy = np.linspace(-1, 1, 300); dd = kde(yy) / kde(yy).max() * 0.40
    ax.fill_betweenx(yy, x0 - dd, x0 + dd, color=col, alpha=0.18, lw=0)
    ax.plot(np.r_[x0 - dd, (x0 + dd)[::-1], x0 - dd[:1]], np.r_[yy, yy[::-1], yy[:1]], color=col, lw=3)
    xj = x0 + rng.uniform(-1, 1, len(z)) * kde(z) / kde(yy).max() * 0.36; yj = np.clip(z + rng.uniform(-0.03, 0.03, len(z)), -1, 1)
    ax.scatter(xj, yj, s=380, color=col, alpha=0.40, edgecolors="none", zorder=3); ax.scatter(xj, yj, s=6, color="black", zorder=4)
    kx.text(X2P(x0), YN, name, ha="center", va="top", fontsize=px(62), fontweight="bold", color=col, linespacing=1.05)
    kx.text(X2P(x0), YNUM + 10, f"{z.mean():.2f}", ha="center", va="top", fontsize=px(120), fontweight="bold", color=col)
xl, xr = X2P(0.5), X2P(1.5)
kx.plot([xl, xl, xr, xr], [YBR + 30, YBR, YBR, YBR + 30], color=INK, lw=5)
kx.text((xl + xr) / 2, YBR - 8, PT, ha="center", va="bottom", fontsize=px(58), fontweight="bold", color=INK)
ax.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=6)
ax.text(XL[1], 0.025, "chance", ha="right", va="bottom", fontsize=px(58), fontweight="bold", color=INK, zorder=7, bbox=dict(fc="white", ec="none", pad=2, alpha=0.85))
ax.set_xlim(*XL); ax.set_ylim(-1.06, 1.06); ax.set_xticks([])
ax.set_yticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticklabels(["\u22121", "\u22120.5", "0", "0.5", "1"], fontsize=px(54), fontweight="bold", color=INK)
ax.tick_params(axis="y", length=14, width=3, color=INK, pad=8)
ax.set_ylabel("how well the reader's ranking\nmatches the true attention", fontsize=px(56), fontweight="bold", color=INK, labelpad=18, linespacing=1.1)
for k_, s_ in ax.spines.items(): s_.set_visible(k_ == "left")
ax.spines["left"].set_linewidth(3); ax.spines["left"].set_bounds(-1, 1)
KY = YN + 40; kx.scatter([50], [KY], s=380, color=INK, alpha=0.35, edgecolors="none"); kx.scatter([50], [KY], s=6, color="black")
kx.text(80, KY, "= one prompt", ha="left", va="center", fontsize=px(52), fontweight="bold", color=INK)
fig.savefig(OUT, dpi=100, facecolor="white"); plt.close(fig)
im = Image.open(OUT).convert("RGB"); im.save(OUT); im.resize((1800, round(1800 * H / W)), Image.LANCZOS).save(OUT[:-4] + ".jpg", quality=92)
print(OUT, im.size)
