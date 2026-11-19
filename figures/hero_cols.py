"""Shared drawing for the control-family figures in the style of hero_L28 / hero_inject (style only, no data):
chat-bubble columns, per-prompt violins on a log-ratio scale (dots coloured with pal.pct_colour), dashed baseline,
big % and p per column, rendered to PNG with headless Chrome for Testing."""
import base64, html, os, subprocess
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from PIL import Image
from pal import PURPLE, RED, INK, pct_colour

AMB = "#F2B53A"; BLUE = "#1F8BFF"
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
ROBOT = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
MINUS = "\u2212"
FONT = '"Helvetica Neue",Helvetica,Arial,sans-serif'

CSS = f"""*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#fff;font-family:{FONT};color:{INK};padding:20px 26px 10px;position:relative}}
.cols{{display:flex;gap:36px}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:76px;font-weight:800;margin:0 0 12px 4px;white-space:nowrap}}
.ub{{flex:1;background:{BLUE};color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap;overflow:hidden}}
.nol{{background:none!important;border:5px dashed #fff;flex:0 0 132px}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:12px}}
.rob{{width:110px;height:110px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:70px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.vw{{position:relative;margin-top:24px}}
.vio{{display:block}}
.ov{{position:absolute;white-space:nowrap}}
.bl{{font-size:58px;font-weight:700;display:flex;align-items:center;gap:16px}}
.cub{{background:{BLUE};border-radius:26px;padding:12px 14px}}
.cub .var{{color:{INK};font-size:46px;white-space:normal}}
.bl.nar{{flex-direction:column;align-items:flex-start;gap:8px}} .bl.nar .cub{{width:440px}}
.lab{{position:absolute;text-align:center;white-space:nowrap;line-height:1}}
.eff{{font-size:96px;font-weight:800;letter-spacing:-1px}}
.pv{{font-size:56px;margin-top:14px}}
.pv sup{{font-size:36px;vertical-align:22px;line-height:0}}
"""

def pct(v): return (f"+{v:.0f}%" if v >= 0 else f"{MINUS}{abs(v):.0f}%")

def bubble_col(name, col, paras, K, line, qline, ans, line_first=False, nwords=(2, 3, 3, 3)):
    """one chat column: condition name, the user bubble (four paragraphs shortened, [PK] amber, the varying line in a white box), robot + reply"""
    pl = "".join(f'<div class="pl"><b class="{"k" if i + 1 == K else ""}">[P{i + 1}]</b> {html.escape(" ".join(p.split()[:nwords[i]]))} …</div>' for i, p in enumerate(paras))
    var = f'<div class="var" style="color:{col}">{html.escape(line)}</div>' if line else ('<div class="var nol"></div>' if line == "" else "")
    q = f'<div>{html.escape(qline).replace(" + ", "&nbsp;+&nbsp;")}</div>'
    body = (var + "<div>Here are four paragraphs.</div>" + pl + q) if line_first else ("<div>Here are four paragraphs.</div>" + pl + var + q)
    nm = f'<div class="cname" style="color:{col}">{name}</div>' if name else ""
    return f'''<div class="col">{nm}<div class="ub">{body}</div>
  <div class="mrow"><img class="rob" src="data:image/png;base64,{ROBOT}"><div class="mb">{html.escape(ans)}</div></div></div>'''

def ticks_for(lo, hi):
    cand = [0.25, 0.5, 0.75, 1.5, 2, 3, 4, 6]
    return [r for r in cand if lo < np.log2(r) < hi]

def violins(cols, VW, VH, png, gap=36, seed=1, top_band=200, bot_band=200, ticks=None, skip=(0, 0), mean_mark=False, tick_right=None):
    """skip: (from, to) px below the dashed baseline kept free of y labels (the baseline label sits there);
    mean_mark: a black bar across each violin at its mean % change (log2(1 + mean/100))"""
    """cols: list (one per page column; None = empty column) of dict(z = log2(value / baseline) per prompt, mean = mean % change, p = p text).
    Draws violins on one shared log-ratio y axis (labels on the left column only), dots coloured by pct_colour, dashed baseline at 0,
    big % and p above a rising column / below a falling one. Returns layout info in page px (relative to the image top-left)."""
    n = len(cols); CW = (VW - gap * (n - 1)) / n
    fig = plt.figure(figsize=(VW / 72, VH / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, VW)
    allz = np.concatenate([c["z"] for c in cols if c is not None]); lo, hi = min(allz.min(), 0), max(allz.max(), 0)
    ups = [c for c in cols if c is not None and c["mean"] > 0]; dns = [c for c in cols if c is not None and c["mean"] <= 0]
    for c in cols:   # violin outline runs 1.2 bandwidths past the extreme dots
        if c is not None: k_ = gaussian_kde(c["z"]); c["_ext"] = 1.2 * k_.factor * c["z"].std()
    s = VH / (hi - lo) * 0.6
    for _ in range(60):   # px per log2 unit such that every label band fits
        yhi = max([hi + 30 / s] + [c["z"].max() + c["_ext"] + 20 / s for c in cols if c is not None] + [c["z"].max() + c["_ext"] + top_band / s for c in ups])
        ylo = min([lo - 30 / s] + [c["z"].min() - c["_ext"] - 20 / s for c in cols if c is not None] + [c["z"].min() - c["_ext"] - bot_band / s for c in dns])
        s = VH / (yhi - ylo)
    ax.set_ylim(ylo, yhi)
    Y = lambda v: (yhi - v) * s   # data -> px from top
    info = dict(CW=CW, y0=Y(0), Y=Y, cx=[], tops=[], bots=[])
    for j, c in enumerate(cols):
        cx = j * (CW + gap) + CW / 2; info["cx"].append(cx)
        if c is None: info["tops"].append(None); info["bots"].append(None); continue
        z = c["z"]; up = c["mean"] > 0; colr = PURPLE if up else RED
        kde = gaussian_kde(z); bw = kde.factor * z.std(); ys = np.linspace(z.min() - 1.2 * bw, z.max() + 1.2 * bw, 500); d = kde(ys)
        hw = min(0.30 * CW, 330) * d / d.max()
        ax.fill_betweenx(ys, cx - hw, cx + hw, color=colr, alpha=0.22, lw=0, zorder=1)
        ax.plot(np.r_[cx + hw, cx - hw[::-1], cx + hw[:1]], np.r_[ys, ys[::-1], ys[:1]], color=colr, lw=2.5, zorder=2)
        g = np.random.default_rng(seed + j)
        xj = cx + g.uniform(-0.92, 0.92, len(z)) * np.maximum(np.interp(z, ys, hw), 14)
        ax.scatter(xj, z, s=38 ** 2, c=pct_colour(2.0 ** z), alpha=0.45, edgecolors="none", zorder=3)
        ax.scatter(xj, z, s=9, color="black", zorder=4)
        if mean_mark: ax.plot([cx - 0.07 * CW, cx + 0.07 * CW], [np.log2(1 + c["mean"] / 100)] * 2, color=INK, lw=7, solid_capstyle="butt", zorder=7)
        info["tops"].append(Y(ys.max())); info["bots"].append(Y(ys.min()))
    ax.axhline(0, color=INK, lw=5, ls=(0, (4, 3)), zorder=5)
    for r in (ticks if ticks is not None else ticks_for(allz.min() - 0.15, allz.max() + 0.2)):   # shared y axis: labels on the left column only
        yp = Y(np.log2(r))
        if yp < 40 or yp > VH - 40 or info["y0"] + skip[0] - 40 < yp < info["y0"] + skip[1] + 40: continue
        ax.text(tick_right or 8, np.log2(r), pct(100 * (r - 1)), fontsize=58, fontweight="bold", color=INK, ha="right" if tick_right else "left", va="center", zorder=6)   # tick_right: right-align the labels at that x
    ax.axis("off"); fig.savefig(png, dpi=72, facecolor="white"); plt.close(fig)
    return info

SUPB = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "−0123456789")
def p_html(p):
    """pfmt text -> HTML with a real superscript: 'p < 10⁻⁵' -> 'p &lt; 10<sup>−5</sup>'"""
    import re
    return re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10<sup>" + m.group(1).translate(SUPB) + "</sup>", html.escape(p))

def labels_html(info, cols, word=" attention"):
    """big mean % and p per column, above the highest dot of a rising column, below the lowest dot of a falling one"""
    o = []
    for cx, top, bot, c in zip(info["cx"], info["tops"], info["bots"], cols):
        if c is None: continue
        up = c["mean"] > 0; colr = PURPLE if up else RED
        eff = f'<div class="eff">{pct(c["mean"])}{word}</div>'; pp = f'<div class="pv">{p_html(c["p"])}</div>'
        if up: o.append(f'<div class="lab" style="left:{cx:.0f}px;top:{top - 26:.0f}px;transform:translate(-50%,-100%);color:{colr}">{eff}{pp}</div>')
        else: o.append(f'<div class="lab" style="left:{cx:.0f}px;top:{bot + 26:.0f}px;transform:translate(-50%,0);color:{colr}">{eff}{pp}</div>')
    return "".join(o)

def key_svg(col=PURPLE):
    r = 30; dx = dy = r + 4; ww = dx + r + 16 + 470
    return (f'<svg width="{ww}" height="{2 * r + 8}" viewBox="0 0 {ww} {2 * r + 8}"><circle cx="{dx}" cy="{dy}" r="{r}" fill="{col}" fill-opacity=".75"/>'
            f'<circle cx="{dx}" cy="{dy}" r="4" fill="#000"/><text x="{dx + r + 16}" y="{dy + 19}" font-size="54" font-weight="800" fill="{INK}" font-family=\'{FONT}\'>= one prompt</text></svg>')

def render(page_html, W, out_dir, out, tmp=None, jpg=True):
    """headless Chrome for Testing -> crop to content -> OUT.png (+ .jpg)"""
    tmp = tmp or out_dir
    hp = os.path.join(tmp, f"_{out}.html"); raw = os.path.join(tmp, f"_{out}_raw.png"); open(hp, "w").write(page_html)
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},4000",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(out_dir, out + ".png"))
    if jpg: im.save(os.path.join(out_dir, out + ".jpg"), quality=92)
    os.remove(hp); os.remove(raw)
    print(out, im.size, "aspect", round(im.size[1] / im.size[0], 2))
    return im.size

def b64(path): return base64.b64encode(open(path, "rb").read()).decode()
