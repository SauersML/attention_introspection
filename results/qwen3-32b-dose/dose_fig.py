import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import colors
sys.path.insert(0, "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro"); import pal
MN = lambda t: t.replace("-", "\u2212")
D = sys.argv[1]; OUT = sys.argv[2]
R = json.load(open(D + "/dose_results.json"))
BAND = list(range(21, 54))   # the band every headline number uses
DJ = json.load(open(D + "/dose.json")); BY = {(r["id"], r["cond"]): r for r in DJ}; IDS = sorted({r["id"] for r in DJ})
def att(c, l): return np.array([BY[i, c]["att_mean"][l][BY[i, c]["k"]] for i in IDS])
def attb(c): return np.array([np.mean([BY[i, c]["att_mean"][l][BY[i, c]["k"]] for l in BAND]) for i in IDS])
def zl(c, ctrl="1_mention", l=None):
    if l is None: return 100 * (attb(c) / attb(ctrl) - 1)
    return 100 * (att(c, l) / att(ctrl, l) - 1)
NL = len(DJ[0]["att_mean"])
INK = pal.INK
plt.rcParams.update({"font.family": "Helvetica", "font.size": 56, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
CS = [("2_glance", "“glance\nat”"), ("3_focus", "“focus\non”"), ("4_focus_strongly", "“focus\nstrongly”"), ("5_very_strongly_only", "“very\nstrongly\nand only”"),
      ("7_little_less", "“a little\nless”"), ("8_do_not", "“do not\npay\nattention”"), ("9_completely_ignore", "“com-\npletely\nignore”")]
CS = sorted(CS, key=lambda c: -zl(c[0]).mean())
Z = [np.log2(1 + zl(c) / 100) for c, _ in CS]  # log ratio at layer LAY; labels in %
allz = np.concatenate(Z); lo_, hi_ = np.percentile(allz, 0.5), np.percentile(allz, 99.5)
norm = pal.norm(np.percentile(allz, 3), np.percentile(allz, 90)); cmap = pal.DIV
fig, a = plt.subplots(figsize=(32, 24)); rng = np.random.default_rng(1)
for p, ((c, lab), z) in enumerate(zip(CS, Z)):
    col = pal.PURPLE if zl(c).mean() > 0 else pal.RED
    v = a.violinplot([z], positions=[p], widths=0.9, showextrema=False)
    for b in v["bodies"]: b.set_facecolor(col); b.set_alpha(0.2); b.set_edgecolor(col); b.set_linewidth(2.5)
    xj = p + rng.uniform(-0.3, 0.3, len(z))
    a.scatter(xj, z, s=420, c=cmap(norm(z)), alpha=0.45, edgecolors="none", zorder=3)
    mp = zl(c).mean()
    up = mp > 0
    a.text(p, (z.max() + 0.05) if up else (z.min() - 0.05), MN(f"{mp:+.0f}%"), fontsize=74, fontweight="bold", ha="center", va="bottom" if up else "top", zorder=7)
a.axhline(0, color=INK, lw=4, zorder=4)
nm = zl("0_none").mean()
a.axhline(np.log2(1 + nm / 100), color=INK, lw=5, ls=(0, (6, 5)), zorder=4)
a.text(-0.45, np.log2(1 + nm / 100) - 0.04, MN(f"no instruction at all ({nm:+.0f}%)"), fontsize=60, fontweight="bold", ha="left", va="top", zorder=8)
a.set_xticks(range(len(CS))); a.set_xticklabels([l for _, l in CS], fontsize=50, fontweight="bold", linespacing=1.05)
for t, (c, _) in zip(a.get_xticklabels(), CS): t.set_color(pal.PURPLE if zl(c).mean() > 0 else pal.RED)
a.set_xlim(-0.6, len(CS) - 0.4); a.set_ylim(allz.min() - 0.35, allz.max() + 0.45)
yt = [-50, 0, 100, 300, 700]
yt = [t for t in yt if allz.min() - 0.35 <= np.log2(1 + t / 100) <= allz.max() + 0.45]
a.set_yticks([np.log2(1 + t / 100) for t in yt]); a.set_yticklabels([MN(f"{t:+d}%") if t else "0%" for t in yt])
a.set_ylabel("change in attention vs just\nmentioning the paragraph name (log scale)", fontsize=74, labelpad=24)
LC = {}
ins = a.inset_axes([0.60, 0.62, 0.39, 0.36])
UPC = ["#4C1D95", "#7C3AED", "#2563EB", "#0D9488"]; DNC = ["#EA580C", "#DC2626", "#7F1D1D"]
ups = [c for c, _ in CS if zl(c).mean() > 0 and not c.startswith("7") and not c.startswith("8")]; dns = [c for c, _ in CS if c not in ups]
for cs_, pal_ in [(ups, UPC), (dns, DNC)]:
    for c, col in zip(cs_, pal_):
        ins.plot(range(NL), [zl(c, l=l).mean() for l in range(NL)], color=col, lw=5)
        LC[c] = col
ins.axhline(0, color=INK, lw=2.5)
yl = ins.get_ylim(); top = yl[1] + 0.06 * (yl[1] - yl[0]); ins.set_ylim(yl[0], yl[1] + 0.55 * (yl[1] - yl[0]))
ins.plot([21, 21, 53, 53], [top - 0.04 * (yl[1] - yl[0]), top, top, top - 0.04 * (yl[1] - yl[0])], color=INK, lw=4, clip_on=False)
ins.text(37, top + 0.03 * (yl[1] - yl[0]), "averaged across\nthese layers", fontsize=42, linespacing=1.0, fontweight="bold", ha="center", va="bottom")
for t, (c, _) in zip(a.get_xticklabels(), CS): t.set_color(LC[c])
ins.set_xticks([0, 63]); ins.set_yticks([]); ins.tick_params(labelsize=44, length=8, width=2.5)
ins.set_xlabel("layer", fontsize=48, labelpad=-30)
for s_ in ins.spines.values(): s_.set_linewidth(4); s_.set_color("black")
for s in ["top", "right"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(2.5); a.spines["bottom"].set_linewidth(2.5); a.tick_params(width=3, length=14, labelsize=84); a.tick_params(axis="x", length=0, labelsize=48)
fig.tight_layout(); fig.savefig(OUT + "dose.png", dpi=100, facecolor="white"); fig.savefig(OUT + "dose.jpg", dpi=100, facecolor="white")
