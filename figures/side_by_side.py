"""Introspection hero (left) and control hero (right) side by side, white gap. Built from the narrower, taller versions rendered for the pair
(hero_L28.py --W=1800 --VH=17.5 -> hero_L28_tall.png; hero_inject.py --W=1800 --VH=1295 -> hero_inject_tall.png), which come out the
same height; any small difference is evened out by scaling both to the taller one.
The default pair always re-renders both halves first, and refuses a half whose stamp (stamp.py) does not match the current script and
PNG: no stale half can be reused.
Above each column: a horizontal curly brace spanning the whole column (tips down, centre point up), drawn as a filled calligraphic stroke
(thin at the tips and the centre point, thick in between), with "attention control" / "attention introspection" centred above it."""
import os, sys, subprocess
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
H = os.path.dirname(os.path.abspath(__file__))
INK = "#1D1D1F"
sys.path.insert(0, H); import stamp
FILES = sys.argv[1:3] if len(sys.argv) > 3 else ("hero_inject_tall.png", "hero_L28_tall.png")   # introspection left, control right
HALVES = {"hero_inject_tall.png": ("hero_inject.py", "--W=1800", "--VH=1295"), "hero_L28_tall.png": ("hero_L28.py", "--W=1800", "--VH=17.5")}
for f in FILES:
    if f in HALVES:   # rebuild from the current script, then check its stamp
        sc, *args = HALVES[f]
        subprocess.run([sys.executable, os.path.join(H, sc), *args], check=True, cwd=H, env={**os.environ, "OMP_NUM_THREADS": "2"})
        stamp.verify(os.path.join(H, f), sc); print(f, "fresh")
LAB = [("attention introspection" if "inject" in f else "attention control") for f in FILES]
a, b = (Image.open(os.path.join(H, f)).convert("RGB") for f in FILES)
OUT = sys.argv[3] if len(sys.argv) > 3 else "hero_pair"
def _top(im): return int(np.where((np.asarray(im) < 250).any(2).any(1))[0][0])
_tr = max(min(_top(a), _top(b)) - 4, 0); a, b = a.crop((0, _tr, a.width, a.height)), b.crop((0, _tr, b.width, b.height))   # no white between the braces and the columns
h = max(a.height, b.height)
a, b = (im if im.height == h else im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in (a, b))
gap = 80; Wt = a.width + gap + b.width
BAND = 395   # px for label + brace

def brace(x0, x1, yb, ht, wmax=26.0, wtip=3.0, n=600):
    """classic typeset curly brace, horizontal, tips down at y=yb, sharp centre point up at y=yb+ht (y up). Each arm: a quarter
    ellipse from the tip (vertical tangent) to the arm's level, a straight run, and a quarter ellipse up into the centre point
    (vertical tangent), so the two mirrored S-curves meet in a point. Drawn as a filled stroke whose width follows sin(pi*s)
    along each arm's arc length: thin at the tip and the point, thickest mid-arm."""
    xm = (x0 + x1) / 2; r = min(ht * 1.1, (xm - x0) * 0.18); hh = ht / 2
    t = np.linspace(0, np.pi / 2, n // 4)
    c1 = np.c_[x0 + r - r * np.cos(t), yb + hh * np.sin(t)]            # tip -> arm level
    ln = np.c_[np.linspace(x0 + r, xm - r, n // 2), np.full(n // 2, yb + hh)]
    c2 = np.c_[xm - r + r * np.sin(t), yb + hh + hh * (1 - np.cos(t))]  # arm level -> centre point
    arm = np.r_[c1, ln[1:], c2[1:]]
    sl = np.r_[0, np.cumsum(np.hypot(*np.diff(arm, axis=0).T))]; sl /= sl[-1]
    w = wtip + (wmax - wtip) * np.sin(np.pi * sl) ** 0.8
    def outline(P, w):
        d = np.gradient(P, axis=0); nrm = np.c_[-d[:, 1], d[:, 0]] / np.hypot(d[:, 0], d[:, 1])[:, None]
        return P + nrm * w[:, None] / 2, P - nrm * w[:, None] / 2
    L1, L2 = outline(arm, w)
    rarm = np.c_[2 * xm - arm[::-1, 0], arm[::-1, 1]]; R1, R2 = outline(rarm, w[::-1])
    return np.r_[L1, R1, R2[::-1], L2[::-1]]   # left outer, right outer, back along right inner and left inner

# label size: as big as both fit over their columns (one shared size), at most 138 pt = 1.5x the heroes' big % labels (89 pt)
_f = plt.figure(dpi=100); _r = _f.canvas.get_renderer()
FS = min([138] + [138 * (w - 40) / _f.text(0, 0, l, fontsize=138, fontweight="bold", family="Helvetica").get_window_extent(_r).width
                  for w, l in ((a.width - 60, LAB[0]), (b.width - 60, LAB[1]))]); plt.close(_f); print("label pt", round(FS))
fig = plt.figure(figsize=(Wt / 100, BAND / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wt); ax.set_ylim(0, BAND); ax.axis("off")
for (x0, x1), lab in (((30, a.width - 30), LAB[0]), ((a.width + gap + 30, Wt - 30), LAB[1])):
    ax.add_patch(Polygon(brace(x0, x1, 4, 116), closed=True, fc=INK, ec=INK, lw=1.0, joinstyle="miter"))
    t = ax.text((x0 + x1) / 2, 118, lab, ha="center", va="bottom", fontsize=FS, fontweight="bold", color=INK, family="Helvetica")
bp = os.path.join(H, "_braces.png"); fig.savefig(bp, dpi=100, facecolor="white"); plt.close(fig)
band = Image.open(bp).convert("RGB"); os.remove(bp)
_rows = np.where((np.asarray(band) < 250).any(2).any(1))[0]; band = band.crop((0, max(_rows[0] - 14, 0), Wt, BAND)); BAND = band.height   # no white above the labels
out = Image.new("RGB", (Wt, BAND + h), "white"); out.paste(band, (0, 0)); out.paste(a, (0, BAND)); out.paste(b, (a.width + gap, BAND))
out.save(os.path.join(H, OUT + ".png")); out.save(os.path.join(H, OUT + ".jpg"), quality=90)
print(out.size, "width/height", round(out.width / out.height, 2))
