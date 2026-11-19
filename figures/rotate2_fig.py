"""rotate2.png: the rotation experiment drawn with its method. Top: the real prompt of context 0 (paragraphs cut to three words),
its instruction line shown as the three alternatives (mention, focus, completely ignore; [Pk] names one paragraph, rotated over all
four), and Qwen3-32B's answer. Below: two 4x4 heatmaps (focus | completely ignore) of the % change in attention from the answer
tokens to each paragraph (layers 21-53) against the mention-only line naming the same paragraph (cell colour = pct_colour of 1 + change/100, the fixed scale of every
figure; the 95% bootstrap CIs stay in rotate2_tables.md), above each map the share of (prompt, named paragraph) pairs where the named
paragraph moved most in the instructed direction (share_named_most; chance 25%), and one shared colour key. Data: results/qwen3-32b-rotate2/rotate2_results.json
(rotate2_analyze.py) and R2.json (the example prompt). Usage: python rotate2_fig.py [rotate2_results.json]"""
import json, sys, os, base64, html, subprocess, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import DIV, PCTNORM, PURPLE, RED, INK
D = os.path.join(HERE, "..", "results", "qwen3-32b-rotate2")
O = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "rotate2_results.json")))
meta = json.load(open(os.path.join(D, "meta.json"))); LN = {c: t.replace("{k}", "k") for c, t in meta["lines"].items()}
MN = lambda s: s.replace("-", "\u2212")
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
P = {c: np.array(O[c]["mean"]) for c in ["focus", "ignore"]}
L2 = {c: np.log2(1 + P[c] / 100) for c in P}   # cell colour = pct_colour(1 + change/100): the fixed ¼×..4× scale of every figure
SH = {c: O[c]["share_named_most"] for c in P}   # share of (prompt, named paragraph) pairs where the named paragraph moved most
W = 2248
fig = plt.figure(figsize=(W / 100, 15.6))
axs = [fig.add_axes([0.085, 0.165, 0.405, 0.70]), fig.add_axes([0.59, 0.165, 0.405, 0.70])]
fmt = lambda v: MN(f"{v:+.0f}%") if round(v) else "0%"
for a, c, title, col, verb in zip(axs, ["focus", "ignore"], ["focus your attention", "completely ignore"], [PURPLE, RED], ["gains", "loses"]):
    a.imshow(L2[c], cmap=DIV, norm=PCTNORM, aspect="auto")
    for k in range(4):
        for m in range(4):
            v = P[c][k, m]; tc = "white" if abs(L2[c][k, m]) > 1.1 else INK
            if k == m: a.text(m, k, fmt(v), ha="center", va="center", fontsize=56, fontweight="bold", color=tc)   # only the named cell; colour shows the rest
        a.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, lw=8, ec=INK, zorder=5))
    a.set_xticks(range(4)); a.set_xticklabels([f"P{i}" for i in range(1, 5)]); a.set_yticks(range(4)); a.set_yticklabels([f"P{i}" for i in range(1, 5)])
    a.xaxis.tick_top(); a.xaxis.set_label_position("top"); a.tick_params(length=0, pad=8, labelsize=40)
    for t in a.get_xticklabels() + a.get_yticklabels(): t.set_fontweight("bold")
    a.set_xlabel("attention to paragraph", fontsize=36, fontweight="bold", labelpad=8)
    a.set_ylabel("paragraph named in the line", fontsize=34, fontweight="bold", labelpad=8)
    for s_ in a.spines.values(): s_.set_visible(False)
    x0 = a.get_position().x0; xc = x0 + a.get_position().width / 2
    fig.text(xc, 0.995, title, ha="center", va="top", fontsize=62, fontweight="bold", color=col)

# one shared colour key, on the same fixed scale as the cells
cax = fig.add_axes([0.3, 0.045, 0.4, 0.045])
gx = np.linspace(-2, 2, 512)[None, :]; cax.imshow(gx, cmap=DIV, norm=PCTNORM, aspect="auto", extent=(-2, 2, 0, 1))
cax.set_yticks([]); tk = [-2, -1, 0, 1, 2]; cax.set_xticks(tk)
cax.set_xticklabels([MN(f"{100 * (2.0 ** t - 1):+.0f}%") if t else "0%" for t in tk]); cax.tick_params(labelsize=40, length=0, pad=8)
for t in cax.get_xticklabels(): t.set_fontweight("bold")
for s_ in cax.spines.values(): s_.set_visible(False)
fig.text(0.29, 0.0675, "less attention", ha="right", va="center", fontsize=42, fontweight="bold", color=RED)
fig.text(0.71, 0.0675, "more attention", ha="left", va="center", fontsize=42, fontweight="bold", color=PURPLE)
hm = os.path.join(HERE, "_rotate2_maps.png"); fig.savefig(hm, dpi=100, facecolor="white"); plt.close(fig)
# ---- method on top: the real prompt of context 0
R = json.load(open(os.path.join(D, "R2.json"))); r = R[0]
u = r["user_example"]; parts = u.split("\n\n"); paras = parts[1:5]; q = parts[5].strip()
assert q.startswith("Unrelated question:")
def para(p):
    lab, txt = p.split("] ", 1); return f'<div><b>{html.escape(lab)}]</b> {html.escape(" ".join(txt.split()[:3]))} …</div>'
def chip(t, col, note=""):
    return f'<div class="vrow"><div class="var" style="color:{col}">{html.escape(t)}</div>{note}</div>'
AMB = "#F2B53A"
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
maps = base64.b64encode(open(hm, "rb").read()).decode(); os.remove(hm)
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:2300px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:20px 28px;font-size:48px;line-height:1.2;display:flex;flex-direction:column;gap:8px}}
.alt{{display:flex;flex-direction:column;gap:10px;border:5px dashed #fff;border-radius:22px;padding:12px;margin:4px 0}}
.var{{background:#fff;border-radius:16px;padding:6px 16px;font-weight:800;font-size:48px;white-space:nowrap}}
.vrow{{display:flex;align-items:center;gap:20px}} .lab{{font-size:46px;font-weight:800;color:#fff;white-space:nowrap}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:10px}} .rob{{width:96px;height:96px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:8px 28px;font-size:54px;font-weight:700}}
.ref{{font-size:46px;font-weight:700;line-height:1.15;margin-left:20px}}
.maps{{width:{W}px;display:block;margin-top:6px}}
</style></head><body>
<div class="ub"><div>{html.escape(parts[0].strip())}</div>{"".join(para(p) for p in paras)}
<div class="alt">{chip(LN["mention"], INK, '<div class="lab">← 0%</div>')}{chip(LN["focus"], PURPLE)}{chip(LN["ignore"], RED, '<div class="lab">[Pk] = P1, P2, P3 or P4</div>')}</div>
<div>{html.escape(q)}</div></div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb"><b>Qwen3-32B:</b> {html.escape(r["ans_none"].strip())}</div>
<div class="ref">% = change in attention from the answer to each paragraph,<br>compared to the same prompt with the “Note that …” line</div></div>
<img class="maps" src="data:image/png;base64,{maps}"></body></html>'''
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
hp = os.path.join(HERE, "rotate2.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "rotate2_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=2300,4000",
                "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, 2300, nz[-1] + 20)); os.remove(raw); os.remove(hp)
im.save(os.path.join(HERE, "rotate2.png")); im.save(os.path.join(HERE, "rotate2.jpg"), quality=92); print(im.size, round(im.size[1] / im.size[0], 2))
