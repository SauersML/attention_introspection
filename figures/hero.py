# Hero figure: real per-token attention on the named paragraph (median context) under
# "focus your attention on [Pk]" and "do not pay attention to [Pk]", above the per-question
# violin of % change vs "only mentioned" (same as combo.py's left panel), Qwen3-32B, arithmetic task.
import json, math, os, subprocess, html, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
from matplotlib import colors
R = "/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/"
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
TOK = "/Users/user/mpd-data/runpod/hypha-prod-attntok2/out/tokens_ctx488.json"
CHROME = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
BLUE = "#2B5FA8"; INK = "#1D1D1F"
def band(att, L): return np.array(att)[L].mean(0)
def boot(x, NB=10000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); i = rng.integers(0, len(x), (NB, len(x))); s = x[i].mean(1)
    return x.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)

T = json.load(open(TOK)); CID = T["id"]; K = T["k"]
# ---------- violin (combo.py left panel, rebuilt from B.json) ----------
meta = json.load(open(R + "qwen3-32b/meta.json")); B = json.load(open(R + "qwen3-32b/B.json")); L = meta["LAYERS"]
res = {}
for name, cond in [("“focus your\nattention”", "focus"), ("“do not pay\nattention”", "ignore")]:
    by = {c: {r["id"]: r for r in B if r["task"] == "arith" and r["cond"] == c} for c in [cond, "mention"]}
    ids = sorted(by[cond]); assert sorted(by["mention"]) == ids
    assert all(by[cond][i]["k"] == by["mention"][i]["k"] for i in ids)
    x = np.array([band(by["mention"][i]["att_mean"], L)[by["mention"][i]["k"]] for i in ids])
    y = np.array([band(by[cond][i]["att_mean"], L)[by[cond][i]["k"]] for i in ids])
    z = 100 * (y / x - 1); mm, lo, hi = boot(z)
    res[name] = dict(z=z, mean=mm, lo=lo, hi=hi, cond=cond, ex=z[ids.index(CID)], ix=ids.index(CID))
    print(cond, "n=%d mean %+.2f%% CI [%.2f, %.2f]; context %d: %+.2f%%" % (len(z), mm, lo, hi, CID, z[ids.index(CID)]))
plt.rcParams.update({"font.family": "Helvetica", "font.size": 56, "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
fig, a = plt.subplots(figsize=(26, 14))
cmap = colors.LinearSegmentedColormap.from_list("hy", ["#BFD3EE", "#5B8BD0", "#2B4FA8", "#4B2A8F", "#2A0F4F"])
allz = np.concatenate([res[n]["z"] for n in res]); vm = np.percentile(np.abs(allz), 97); norm = colors.Normalize(vmin=0, vmax=vm)
rng = np.random.default_rng(1)
for p, name in enumerate(res):
    r = res[name]; z = r["z"]
    vv = a.violinplot([z], positions=[p * 1.35], widths=0.95, showextrema=False)
    for bb in vv["bodies"]: bb.set_facecolor(BLUE); bb.set_alpha(0.22); bb.set_edgecolor(BLUE); bb.set_linewidth(2.5)
    xj = p * 1.35 + rng.uniform(-0.34, 0.34, len(z)); a.scatter(xj, z, s=650, c=cmap(norm(np.abs(z))), alpha=0.45, edgecolors="none", zorder=3); a.scatter(xj, z, s=6, color="black", zorder=4)
    a.plot([p * 1.35, p * 1.35], [r["lo"], r["hi"]], color=INK, lw=12, solid_capstyle="butt", zorder=5); a.plot([p * 1.35 - 0.22, p * 1.35 + 0.22], [r["mean"]] * 2, color=INK, lw=6, zorder=6)
    a.scatter([xj[r["ix"]]], [z[r["ix"]]], s=1100, facecolors="none", edgecolors=INK, linewidths=5, zorder=8)
    a.text(p * 1.35, (z.max() + 2) if r["mean"] > 0 else (z.min() - 2), f"{r['mean']:+.0f}%", fontsize=90, fontweight="bold", ha="center", va=("bottom" if r["mean"] > 0 else "top"), zorder=7)
a.set_xticks([0, 1.35]); a.set_xticklabels(list(res), fontsize=60, fontweight="bold", linespacing=1.1); a.set_xlim(-0.75, 2.1)
a.text(0.99, 0.97, "Qwen3-32B", transform=a.transAxes, ha="right", va="top", fontsize=64, fontweight="bold")
a.set_ylim(allz.min() - 17, allz.max() + 16)
a.axhline(0, color=INK, lw=4, zorder=4)
a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%"))
for s_ in ["top", "right"]: a.spines[s_].set_visible(False)
a.spines["left"].set_linewidth(3); a.spines["bottom"].set_linewidth(3); a.tick_params(width=4, length=16, labelsize=72)
a.set_ylabel("change in attention\nvs only mentioned", fontsize=64, labelpad=10)
fig.tight_layout(pad=0.6); fig.savefig(OUT + "hero_violin.png", dpi=110, facecolor="white"); plt.close(fig)

# ---------- token rows ----------
rows = [("focus", "“focus your attention on [P%d]”" % (K + 1)), ("ignore", "“do not pay attention to [P%d]”" % (K + 1))]
P = {c: T["conds"][c]["paras"][K] for c, _ in rows}
assert P["focus"]["tokens"] == P["ignore"]["tokens"]
vals = np.concatenate([P[c]["att"] for c, _ in rows]); vmin, vmax = vals.min(), vals.max()
print("token attention range %.3g .. %.3g (x%.0f)" % (vmin, vmax, vmax / vmin))
LOG = vmax / vmin > 20
def f(v): return (math.log(v / vmin) / math.log(vmax / vmin)) if LOG else (v - vmin) / (vmax - vmin)
tcm = colors.LinearSegmentedColormap.from_list("pur", ["#F6F2FD", "#D9C8F7", "#A57BEB", "#7B3FE4", "#4A1C9E", "#250A55"])
def tile(t, v):
    u = f(v); bg = colors.to_hex(tcm(u)); fg = "#FFFFFF" if u > 0.5 else INK
    sp = t.startswith(" ") or t.startswith("\n")
    return f'<span class="t{" s" if sp else ""}" style="background:{bg};color:{fg}">{html.escape(t.strip()) or "&nbsp;"}</span>'
# the paragraph's first word shares a token with the space after the label, so it is outside the measured span: shown untinted
FIRST = next(r for r in B if r["id"] == CID and r["cond"] == "focus")["user"].split("[P%d] " % (K + 1))[1].split()[0]
assert not P["focus"]["tokens"][0].strip().startswith(FIRST), FIRST
N = len(P["focus"]["tokens"]); SHOW = N if N <= 110 else 60
blocks = ""
for c, lab in rows:
    toks = "".join(tile(t, v) for t, v in zip(P[c]["tokens"][:SHOW], P[c]["att"][:SHOW]))
    if SHOW < N: toks += '<span class="t s">…</span>'
    blocks += f'<div class="lab">{lab}</div><div class="para"><span class="t pl">{html.escape(FIRST)}</span>{toks}</div>'
stops = ",".join(f"{colors.to_hex(tcm(i / 20))} {i * 5}%" for i in range(21))
page = f"""<html><head><style>
body{{margin:0;background:#fff;font-family:Helvetica,Arial,sans-serif;color:{INK};width:2400px}}
.wrap{{padding:40px 50px 30px 50px}}
.lab{{font-size:58px;font-weight:bold;margin:34px 0 16px 0}}
.lab:first-child{{margin-top:0}}
.para{{display:flex;flex-wrap:wrap;row-gap:10px;font-size:38px;line-height:1}}
.t{{padding:9px 3px;border-radius:4px}}
.s{{margin-left:10px}}
.pl{{background:#fff;color:{INK}}}
.cbh{{display:flex;align-items:center;justify-content:flex-end;margin-top:30px;font-size:52px;font-weight:bold}}
.hbar{{width:900px;height:60px;border-radius:8px;margin:0 22px;background:linear-gradient(to right,{stops})}}
.bot{{display:flex;align-items:center;margin-top:40px}}
.bot img{{width:2300px}}
.cb{{margin-left:120px;display:flex;flex-direction:column;align-items:center}}
.bar{{width:120px;height:760px;border-radius:8px;background:linear-gradient(to top,{stops})}}
.cbt{{font-size:52px;font-weight:bold;margin:14px 0}}
.cbl{{font-size:64px;font-weight:bold;margin-top:18px}}
</style></head><body><div class="wrap">{blocks}
<div class="cbh"><span>less</span><div class="hbar"></div><span>more attention</span></div>
<div class="bot"><img src="hero_violin.png"></div>
</div></body></html>"""
open(OUT + "hero.html", "w").write(page)
subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1", "--window-size=2400,3000",
                "--screenshot=" + OUT + "hero_raw.png", "file://" + OUT + "hero.html"], check=True, capture_output=True)
from PIL import Image
im = Image.open(OUT + "hero_raw.png").convert("RGB"); px = np.asarray(im)
nz = np.where((px < 250).any(2).any(1))[0]; im = im.crop((0, 0, im.width, min(im.height, nz.max() + 40)))
im.save(OUT + "hero.png"); im.save(OUT + "hero.jpg", quality=92)
print("saved", OUT + "hero.png", im.size)
