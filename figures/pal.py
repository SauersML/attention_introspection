"""Shared colours for every attention figure: above zero blue to purple, below zero orange to red (hue shifts with size)."""
from matplotlib import colors
PURPLE = "#5E2B97"; RED = "#C0392B"; INK = "#1D1D1F"
# hue shifts with size: small drops orange, large drops red; small rises blue, large rises purple
DIV = colors.LinearSegmentedColormap.from_list("div", [(0, "#8E1230"), (0.2, RED), (0.4, "#EE8A3C"), (0.5, "#F1ECE4"), (0.6, "#5FA4DC"), (0.8, "#7A55C9"), (1, "#3E0F73")])
def norm(lo, hi):
    """centre the diverging map on 0 whatever the range"""
    lo = min(lo, -1e-9); hi = max(hi, 1e-9)
    return colors.TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
# One fixed colour per % change, the same in every figure: log2(value / baseline), symmetric, clipped at the ends.
# Range ±PCTR in log2: ratio 1/2**PCTR (darkest red) .. 2**PCTR (darkest purple); 1 = no change (cream).
import numpy as _np
PCTR = 2.0
PCTNORM = colors.Normalize(vmin=-PCTR, vmax=PCTR, clip=True)
def pct_colour(ratio):
    """ratio = value / baseline (1 = no change); 0 or below takes the darkest red end"""
    r = _np.asarray(ratio, dtype=float)
    return DIV(PCTNORM(_np.log2(_np.clip(r, 2.0 ** -PCTR, 2.0 ** PCTR))))
