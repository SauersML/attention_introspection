"""dose.png: the wording ladder. Per prompt (150 arithmetic prompts, Qwen3-32B), attention from the answer tokens to the named
paragraph, averaged over layers 21-53, as % change against the same prompt whose line only mentions the paragraph
("Note that one of the paragraphs is labelled [Pk]."). Everything is read from dose_results.json: per-prompt values
(vs_mention[c]["z"]), the mean with its 95% bootstrap CI (vs_mention[c]["mean"] = [mean, lo, hi]) and the per-layer means
for the inset (per_layer_vs_mention). Linear axis. Usage: python dose_fig.py RESULTS_DIR OUT_PREFIX"""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")); import pal
MN = lambda t: t.replace("-", "\u2212")
D = sys.argv[1]; OUT = sys.argv[2]
R = json.load(open(os.path.join(D, "dose_results.json"))); VM = R["vs_mention"]; PL = R["per_layer_vs_mention"]
B0, B1 = R["layers"]
INK = pal.INK; DOT = "#3A3A40"
FS = 48
plt.rcParams.update({"font.family": "Helvetica", "font.size": FS, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
CS = [("2_glance", "“glance\nat”"), ("3_focus", "“focus\nyour\nattention\non”"), ("4_focus_strongly", "“focus\nstrongly\non”"),
      ("5_very_strongly_only", "“focus\nvery\nstrongly\nand only”"), ("7_little_less", "“a little\nless”"),
      ("8_do_not", "“do not\npay\natten-\ntion”"), ("9_completely_ignore", "“com-\npletely\nignore”")]
CS = sorted(CS, key=lambda c: -VM[c[0]]["mean"][0])
UPC = ["#4C1D95", "#7C3AED", "#2563EB", "#0D9488"]; DNC = ["#EA580C", "#DC2626", "#7F1D1D"]
COL = {}
ups = [c for c, _ in CS if VM[c]["mean"][0] > 5]; dns = [c for c, _ in CS if c not in ups]
for cs_, pl_ in [(ups, UPC), (dns, DNC)]:
    for c, col in zip(cs_, pl_): COL[c] = col
fig, a = plt.subplots(figsize=(22.5, 15.2)); rng = np.random.default_rng(1)
for p, (c, lab) in enumerate(CS):
    z = np.array(VM[c]["z"]); m, lo, hi = VM[c]["mean"]
    v = a.violinplot([z], positions=[p], widths=0.86, showextrema=False)
    for b in v["bodies"]: b.set_facecolor("#E9E9EE"); b.set_alpha(1); b.set_edgecolor("#B9B9C2"); b.set_linewidth(2)
    xj = p + rng.uniform(-0.28, 0.28, len(z)); a.scatter(xj, z, s=110, c=pal.pct_colour(1 + z / 100), alpha=0.4, edgecolors="none", zorder=3); a.scatter(xj, z, s=2, color="black", zorder=4)
    # mean with its 95% CI: a thick bar at the mean, whiskers to the interval
    a.plot([p - 0.3, p + 0.3], [m, m], color=COL[c], lw=9, solid_capstyle="butt", zorder=6)
    a.plot([p, p], [lo, hi], color=COL[c], lw=5, zorder=6)
    for yy in (lo, hi): a.plot([p - 0.12, p + 0.12], [yy, yy], color=COL[c], lw=5, zorder=6)
    up = m > 5
    a.text(p, (z.max() + 3) if up else (z.min() - 3), MN(f"{m:+.0f}%"), fontsize=60, fontweight="bold", color=COL[c], ha="center", va="bottom" if up else "top", zorder=7)
a.axhline(0, color=INK, lw=4, zorder=4)
nm = VM["0_none"]["mean"][0]
a.axhline(nm, color=INK, lw=4, ls=(0, (6, 5)), zorder=4)
a.text(-0.45, nm - 1.5, MN(f"no instruction line at all ({nm:+.0f}%)"), fontsize=FS, fontweight="bold", ha="left", va="top", zorder=8)
a.set_xticks(range(len(CS))); a.set_xticklabels([l for _, l in CS], fontsize=FS, fontweight="bold", linespacing=1.0, va="top")
for t, (c, _) in zip(a.get_xticklabels(), CS): t.set_color(COL[c])
lo_all = min(min(VM[c]["z"]) for c, _ in CS); hi_all = max(max(VM[c]["z"]) for c, _ in CS)
a.set_xlim(-0.55, len(CS) - 0.45); a.set_ylim(lo_all - 16, hi_all + 14)
yt = [-25, 0, 25, 50, 75, 100]; a.set_yticks(yt); a.set_yticklabels([MN(f"{t:+d}%") if t else "0%" for t in yt])
a.set_ylabel("attention to the named paragraph", fontsize=FS, labelpad=8)
# inset: each wording per layer, the averaged band bracketed
ins = a.inset_axes([0.585, 0.585, 0.41, 0.41])
NL = len(PL["3_focus"])
for c, _ in CS: ins.plot(range(NL), PL[c], color=COL[c], lw=5)
ins.axhline(0, color=INK, lw=2.5)
yl = ins.get_ylim(); span = yl[1] - yl[0]; top = yl[1] + 0.06 * span; ins.set_ylim(yl[0], yl[1] + 0.62 * span)
ins.plot([B0, B0, B1, B1], [top - 0.05 * span, top, top, top - 0.05 * span], color=INK, lw=4)
ins.text((B0 + B1) / 2, top + 0.04 * span, "averaged across\nthese layers", fontsize=FS, linespacing=1.0, fontweight="bold", ha="center", va="bottom")
ins.set_xticks([0, NL - 1]); ins.set_yticks([]); ins.tick_params(labelsize=FS, length=8, width=2.5, pad=4)
ins.set_xlabel("layer", fontsize=FS, labelpad=-44)
for s_ in ins.spines.values(): s_.set_linewidth(3.5); s_.set_color(INK)
for s in ["top", "right"]: a.spines[s].set_visible(False)
a.spines["left"].set_linewidth(3); a.spines["bottom"].set_linewidth(3); a.tick_params(width=3, length=12, labelsize=FS); a.tick_params(axis="x", length=0, pad=10)
fig.subplots_adjust(left=0.15, right=0.995, top=0.995, bottom=0.2)
fig.savefig(OUT + "dose.png", dpi=100, facecolor="white"); fig.savefig(OUT + "dose.jpg", dpi=100, facecolor="white")
print({c: np.round(VM[c]["mean"], 1).tolist() for c, _ in CS})
