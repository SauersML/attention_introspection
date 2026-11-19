"""Shared by told_pair.py (paragraphs) and words_told_pair.py (word lists): the sign-flip test, the bootstrap over contexts, and one bar
panel (purple bars, black 95% CI whiskers, dashed 0 line, big purple labels with the raw rates under them, black dose labels for the
measured attention ratio, bold y tick labels and y axis title). Style and statistics only: no data is read here."""
import numpy as np
from pal import INK, PURPLE
from pfmt import pstr
import injb_draw as dr


def signflip(x, NF=200000, seed=0):
    """two-sided sign-flip test of mean(x) = 0 over contexts"""
    x = np.asarray(x); rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1)


def boot(f, N, n=20000, seed=1):
    """95% bootstrap interval of f(index array) over N contexts"""
    g = np.random.default_rng(seed); out = np.empty(n)
    for i in range(n): out[i] = f(g.integers(0, N, N))
    return np.percentile(out, [2.5, 97.5])


rel_lab = lambda mu: f"{100*mu:+.0f}%".replace("-", "\u2212")


def bars_panel(fig, ax, x0, G, MS, bars, base, rate, masses, ticks, item, p=None, NF=200000, ylab="change from no boost",
               zero="0% =\nno boost", tick_lab=lambda t: f"+{100*t:.0f}%", big=rel_lab, base_lab=lambda b: f"{100*b:.0f}%", rate_lab=lambda r: f"{100*r:.0f}%"):
    """one panel at x0 (canvas px) on ax (x in canvas px, y in data units); G: layout (PW, LM, CW, VW, VH, ylo, yhi).
    bars[m] = (height, (lo, hi) or None); base: no-boost rate; rate[m]: boosted rate; ticks: labelled y values; p: p at the top dose
    (None: not drawn); base_lab(base): how the no-boost rate is written; item: 'paragraph' / 'word' in the dose axis title; big(height): the big label over each bar."""
    PW, LM, CW, VW, VH, ylo, yhi = (G[k] for k in ("PW", "LM", "CW", "VW", "VH", "ylo", "yhi"))
    ax.plot([x0 + LM - 20, x0 + PW], [0, 0], color=INK, lw=4, ls=(0, (4, 3)), zorder=2)
    for t in ticks:
        ax.text(x0 + LM - 30, t, tick_lab(t), ha="right", va="center", fontsize=dr.px(72), fontweight="bold", color=INK)
    ax.text(x0 + 30, (ylo + yhi) / 2 + 0.12 * (yhi - ylo) / 1.78, ylab, rotation=90, ha="center", va="center", fontsize=dr.px(64), fontweight="bold", color=INK)
    if zero: ax.text(x0 + LM - 30, 0, zero, ha="right", va="center", fontsize=dr.px(60), fontweight="bold", color=INK, linespacing=1.0)
    for i, m in enumerate(MS):
        xc = x0 + LM + (i + 0.5) * CW; hw = 0.36 * CW; cxf = xc / VW
        mu, ci = bars[m]
        ax.bar(xc, mu, width=2 * hw, bottom=0, color=PURPLE, alpha=0.7, lw=0, zorder=3)
        if ci is not None:
            lo, hi = ci
            ax.plot([xc, xc], [lo, hi], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            for yy in (lo, hi): ax.plot([xc - 0.25 * hw, xc + 0.25 * hw], [yy, yy], color=INK, lw=6, zorder=5, solid_capstyle="butt")
        if p is not None and m == MS[-1]:
            fig.text(cxf, 1 - 178 / VH, pstr(p, floor=1 / (NF + 1)), ha="center", va="top", fontsize=56, color=PURPLE)
        fig.text(cxf, 1 - 6 / VH, big(mu), ha="center", va="top", fontsize=78, fontweight="bold", color=PURPLE)
        fig.text(cxf, 1 - 112 / VH, f"{base_lab(base)} \u2192 {rate_lab(rate[m])}", ha="center", va="top", fontsize=36, fontweight="bold", color=INK, family=["Helvetica", "Arial Unicode MS"])
        fig.text(cxf, 196 / VH, f"\u00d7{masses[m]:.1f}", ha="center", va="center", fontsize=78, fontweight="bold", color=INK)
    fig.text((x0 + LM + 2 * CW) / VW, 52 / VH, f"attention on the boosted {item}", ha="center", va="center", fontsize=dr.px(76), fontweight="bold", color=INK)
