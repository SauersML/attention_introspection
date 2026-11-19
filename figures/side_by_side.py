"""Control hero and introspection hero side by side, white gap. Built from the narrower, taller versions rendered for the pair
(hero_L28.py --W=1800 --VH=17.5 -> hero_L28_tall.png; hero_inject.py --W=1800 --VH=1295 -> hero_inject_tall.png), which come out the
same height; any small difference is evened out by scaling both to the taller one.
Above each column: a horizontal curly brace spanning the whole column (tips down, centre point up), drawn as a filled calligraphic stroke
(thin at the tips and the centre point, thick in between), with "attention control" / "attention introspection" centred above it."""
import os
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
H = os.path.dirname(os.path.abspath(__file__))
INK = "#1D1D1F"
a, b = (Image.open(os.path.join(H, f)).convert("RGB") for f in ("hero_L28_tall.png", "hero_inject_tall.png"))
h = max(a.height, b.height)
a, b = (im if im.height == h else im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in (a, b))
gap = 80; Wt = a.width + gap + b.width
BAND = 250   # px for label + brace

def brace(x0, x1, yb, ht, wmax=17.0, wtip=3.0, n=400):
    """filled brace polygon from x0 to x1, tips at y=yb, centre point at y=yb+ht (y up)."""
    q = np.linspace(0, 1, n); k = 40.0
    sg = lambda t: 1 / (1 + np.exp(-k * t))
    def half(q):   # 0 at the tip, 1 at the centre point: rise near the tip, flat, rise near the centre
        y = 0.5 * (sg(q - 0.06) - sg(-0.06)) / (sg(0.94) - sg(-0.06)) + 0.5 * (sg(q - 0.94) - sg(-0.94)) / (sg(0.06) - sg(-0.94))
        return y
    L = (x1 - x0) / 2
    xs = np.r_[x0 + q * L, x0 + L + q[1:] * L]; ys = np.r_[half(q), half(1 - q[1:])] * ht + yb
    w = np.r_[wtip + (wmax - wtip) * np.sin(np.pi * q) ** 0.6, wtip + (wmax - wtip) * np.sin(np.pi * q[1:]) ** 0.6]
    dx, dy = np.gradient(xs), np.gradient(ys); nn = np.hypot(dx, dy); nx, ny = -dy / nn, dx / nn
    top = np.c_[xs + nx * w / 2, ys + ny * w / 2]; bot = np.c_[xs - nx * w / 2, ys - ny * w / 2]
    return np.r_[top, bot[::-1]]

fig = plt.figure(figsize=(Wt / 100, BAND / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, Wt); ax.set_ylim(0, BAND); ax.axis("off")
for (x0, x1), lab in (((30, a.width - 30), "attention control"), ((a.width + gap + 30, Wt - 30), "attention introspection")):
    ax.add_patch(Polygon(brace(x0, x1, 16, 88), closed=True, fc=INK, ec=INK, lw=1.2, joinstyle="round"))
    ax.text((x0 + x1) / 2, 112, lab, ha="center", va="bottom", fontsize=92 * 0.72 * 1.25, fontweight="bold", color=INK, family="Helvetica")
bp = os.path.join(H, "_braces.png"); fig.savefig(bp, dpi=100, facecolor="white"); plt.close(fig)
band = Image.open(bp).convert("RGB"); os.remove(bp)
out = Image.new("RGB", (Wt, BAND + h), "white"); out.paste(band, (0, 0)); out.paste(a, (0, BAND)); out.paste(b, (a.width + gap, BAND))
out.save(os.path.join(H, "hero_pair.png")); out.save(os.path.join(H, "hero_pair.jpg"), quality=90)
print(out.size, "width/height", round(out.width / out.height, 2))
