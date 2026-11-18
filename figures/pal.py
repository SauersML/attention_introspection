"""Shared colours for every attention figure: above zero = purple, below zero = red, white at zero."""
from matplotlib import colors
PURPLE = "#5E2B97"; RED = "#C0392B"; INK = "#1D1D1F"
DIV = colors.LinearSegmentedColormap.from_list("div", ["#8E1F1F", RED, "#F2B8B0", "#FFFFFF", "#D3C2EC", "#8A5CC7", PURPLE, "#2E0F57"])
def norm(lo, hi):
    """centre the diverging map on 0 whatever the range"""
    lo = min(lo, -1e-9); hi = max(hi, 1e-9)
    return colors.TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
