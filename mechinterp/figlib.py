"""Shared helpers for the mechinterp figures: SVG text/chips and a headless-Chrome render to PNG."""
import os, subprocess, html
import numpy as np
from PIL import Image
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
INK = "#1D1D1F"; PURPLE = "#5E2B97"; RED = "#C0392B"; BLUE = "#2B5FA8"; ORANGE = "#E07B24"; AMB = "#F5B83D"; UB = "#1F8BFF"
FONT = 'Helvetica Neue,Helvetica,Arial,sans-serif'
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

def esc(s): return html.escape(s)

def tw(s, size, bold=False):
    """rough rendered width of Helvetica text"""
    w = 0
    for ch in s:
        if ch in "il.,:;'|![]()1 ": w += 0.30
        elif ch in "mwMW": w += 0.85
        elif ch.isupper() or ch.isdigit(): w += 0.64
        elif ord(ch) > 0x2E80: w += 1.0
        else: w += 0.53
    return w * size * (1.06 if bold else 1.0)

def text(x, y, s, size=46, col=INK, w=400, anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{w}" fill="{col}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'

def chip(x, y, s, size=44, fill="#fff", col=INK, stroke=None, w=400, pad=12, h=None, r=12):
    """token chip with top-left at (x, y); returns (svg, width)"""
    W = tw(s, size, w >= 600) + 2 * pad; H = h or size * 1.35
    st = f' stroke="{stroke}" stroke-width="4"' if stroke else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{W:.1f}" height="{H:.1f}" rx="{r}" fill="{fill}"{st}/>'
            + text(x + W / 2, y + H * 0.72, s, size, col, w, "middle")), W

def render(svg_body, W, H, name):
    svg = f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">{svg_body}</svg>'
    sp = os.path.join(FIG, name + ".svg"); open(sp, "w").write(svg)
    hp = os.path.join(FIG, name + ".html")
    open(hp, "w").write(f'<!doctype html><html><head><meta charset="utf-8"><style>*{{margin:0;padding:0}}body{{background:#fff}}</style></head><body>{svg}</body></html>')
    raw = os.path.join(FIG, name + "_raw.png")
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); os.remove(raw); os.remove(hp)
    out = os.path.join(FIG, name + ".png"); im.save(out); print(name, im.size); return out

def boot_ci(x, nb=4000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (nb, len(x)))].mean(1)
    return x.mean(), np.quantile(m, .025), np.quantile(m, .975)

def signflip_p(z, nf=200000, seed=0):
    rng = np.random.default_rng(seed); z = np.asarray(z, float); obs = abs(z.mean()); hits = 0
    for _ in range(nf // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(z))); hits += int((np.abs((s * z).mean(1)) >= obs).sum())
    return (hits + 1) / (nf + 1)
