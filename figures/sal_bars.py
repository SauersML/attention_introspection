"""Shared pieces for salience.py and pick_random.py: loading results, per-context arms, bootstrap CIs, sign-flip p, and the bar-panel row()."""
import json, glob, os, sys, base64
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from pfmt import pstr
import injb_draw as dr
ROOT = os.path.dirname(HERE)
NF = 20000; NB = 10000


def load(d):
    return sorted((json.loads(l) for f in glob.glob(os.path.join(ROOT, f"results/{d}/data/recs_*.jsonl")) for l in open(f)), key=lambda r: r["id"])


yes = lambda t, k: float(t["text"].strip().startswith("Yes"))
top = lambda t, k: float(int(np.argmax(t["lp_par"])) == k)
patt = lambda t, k: 1 / (1 + np.exp(t["lp_act"] - t["lp_att"]))
def arm(recs, a, f):
    return np.array([f(r[a], 0) if "lp_par" in r[a] else np.mean([f(r[a][str(k)], k) for k in range(4)]) for r in recs])


rng = np.random.default_rng(0)
def ci(v):
    b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1); return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def signflip(d):
    s = rng.choice([-1, 1], (NF, len(d))); null = np.abs((s * d).mean(1)); return float((np.sum(null >= abs(d.mean())) + 1) / (NF + 1))


from salience_draw import GREEN
# ---------- bar panels (one matplotlib image per row) ----------
W = 3000; PAD = 26; VW = W - 2 * PAD
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
NONE = ("none", "white", 1.0, "nothing\nchanged", INK)
LAB = dict(A=(PURPLE, 0.8, "attention\nboost", PURPLE), B=(GREEN, 0.8, "meaning\nadded", GREEN), Bs=(GREEN, 0.8, "meaning\nadded", GREEN),
           QA=(PURPLE, 0.35, "asked to\npick at\nrandom", PURPLE), AQ=(PURPLE, 0.8, "asked\n\u201cwhich\none?\u201d", PURPLE))
FB, FV, FT, FP = 74, 112, 104, 84   # bar labels, bar values, panel titles, p and chance


def row(path, panels, H=1150, BOT=330, TOPH=200, VW_=None, LM0=10):
    """panels: list of (width share, title, ylim top, [(key, values)], chance or None, chance label, [(i, j, p)], gap-after index)"""
    VW = VW_ or globals()["VW"]
    fig = plt.figure(figsize=(VW / 100, H / 100), dpi=100)
    tot = sum(p[0] for p in panels); GAP = 30
    x0 = 0
    for share, title, ytop, bars, chance, clab, brs, sep in panels:
        pw = (VW - GAP * (len(panels) - 1)) * share / tot; LM = LM0; RM = 360 if chance is not None else 10
        ax = fig.add_axes([(x0 + LM) / VW, BOT / H, (pw - LM - RM) / VW, (H - BOT - TOPH) / H])
        nb = len(bars) + (0.5 if sep is not None else 0); ax.set_xlim(-0.5, nb - 0.5); ax.set_ylim(0, ytop)
        xs = [i + (0.5 if sep is not None and i > sep else 0) for i in range(len(bars))]
        hi_top = {}
        for i, ((key, v), xc) in enumerate(zip(bars, xs)):
            if key == "none": col, al, lab, tc = NONE[1], 1.0, NONE[3], INK
            else: col, al, lab, tc = LAB[key]
            mu, lo, hi = (100 * x for x in v)
            ax.bar(xc, mu, width=0.74, color=col, alpha=al, lw=4 if key == "none" else 0, edgecolor=INK, zorder=3)
            ax.plot([xc, xc], [lo, hi], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            for yy in (lo, hi): ax.plot([xc - 0.09, xc + 0.09], [yy, yy], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            ax.text(xc, hi + ytop * 0.015, f"{mu:.0f}%", ha="center", va="bottom", fontsize=dr.px(FV), fontweight="bold", color=tc)
            hi_top[i] = hi + ytop * (0.05 + 1.05 * FV / (H - BOT - TOPH))
            dx = 0.1 * (2 * i - 1) if len(bars) == 2 else 0
            fig.text((x0 + LM + (xc + dx + 0.5) / nb * (pw - LM - RM)) / VW, (BOT - 14) / H, lab, ha="center", va="top", fontsize=dr.px(FB), fontweight="bold", color=tc, linespacing=0.95)
        if chance is not None:
            ax.plot([-0.5, nb - 0.5], [chance, chance], color=INK, lw=4, ls=(0, (4, 3)), zorder=6)
            ax.text(nb - 0.5 + 0.03, chance, clab, ha="left", va="center", linespacing=0.95, fontsize=dr.px(FP), fontweight="bold", color=INK, zorder=7)
        lvl = 0
        for i, j, p in brs:
            yb = max(hi_top[k] for k in range(min(i, j), max(i, j) + 1)) + lvl
            ax.plot([xs[i], xs[i], xs[j], xs[j]], [yb - ytop * 0.03, yb, yb, yb - ytop * 0.03], color=INK, lw=5, zorder=5, clip_on=False)
            ax.text((xs[i] + xs[j]) / 2, yb + ytop * 0.01, pstr(p, floor=1 / (NF + 1)), ha="center", va="bottom", fontsize=dr.px(FP), color=INK)
            lvl += ytop * 0.17
        ax.plot([-0.5, nb - 0.5], [0, 0], color=INK, lw=4, zorder=4)
        fig.text((x0 + LM + (pw - LM - RM) / 2) / VW, (H - 6) / H, title, ha="center", va="top", fontsize=dr.px(FT), fontweight="bold", color=INK, linespacing=1.0)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        x0 += pw + GAP
    fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)

