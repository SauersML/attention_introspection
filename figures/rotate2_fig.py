"""rotate2.png: the rotation experiment drawn with its method. Top: the real prompt of context 0 (paragraphs cut to three words),
its instruction line shown as the three alternatives (mention, focus, completely ignore; [Pk] names one paragraph, rotated over all
four), and Qwen3-32B's answer. Below: two 4x4 heatmaps (focus | completely ignore) of the % change in attention from the answer
tokens to each paragraph (layers 21-53) against the mention-only line naming the same paragraph, with each cell's 95% bootstrap CI
written under its mean, and one shared diverging colour key. Data: results/qwen3-32b-rotate2/rotate2_results.json
(rotate2_analyze.py) and R2.json (the example prompt). Usage: python rotate2_fig.py [rotate2_results.json]"""
import json, sys, os, base64, html, subprocess, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colorbar import ColorbarBase
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import DIV, norm, PURPLE, RED, INK
D = os.path.join(HERE, "..", "results", "qwen3-32b-rotate2")
O = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "rotate2_results.json")))
meta = json.load(open(os.path.join(D, "meta.json"))); LN = {c: t.replace("{k}", "k") for c, t in meta["lines"].items()}
MN = lambda s: s.replace("-", "\u2212")
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})
P = {c: np.array(O[c]["mean"]) for c in ["focus", "ignore"]}
LO = {c: np.array(O[c]["lo"]) for c in P}; HI = {c: np.array(O[c]["hi"]) for c in P}
vm = max(np.abs(P["focus"]).max(), np.abs(P["ignore"]).max()); nm = norm(-vm, vm)
W = 2248
fig = plt.figure(figsize=(W / 100, 13.3))
axs = [fig.add_axes([0.09, 0.175, 0.405, 0.655]), fig.add_axes([0.59, 0.175, 0.405, 0.655])]
fmt = lambda v: MN(f"{v:+.0f}%") if round(v) else "0%"
for a, c, title, col in zip(axs, ["focus", "ignore"], ["“focus your attention”", "“completely ignore”"], [PURPLE, RED]):
    M = P[c]; a.imshow(M, cmap=DIV, norm=nm, aspect="auto")
    for k in range(4):
        for m in range(4):
            v = M[k, m]; dark = abs(v) > 0.45 * vm
            tc = "white" if dark else INK
            a.text(m, k - 0.13, fmt(v), ha="center", va="center", fontsize=56 if k == m else 52, fontweight="bold", color=tc)
            hw = int(np.ceil(max(HI[c][k, m] - v, v - LO[c][k, m])))   # half-width of the 95% CI, rounded up
            a.text(m, k + 0.25, f"±{hw}", ha="center", va="center", fontsize=42, fontweight="bold", color=tc)
        a.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, lw=8, ec=INK, zorder=5))
    a.set_xticks(range(4)); a.set_xticklabels([f"P{i}" for i in range(1, 5)]); a.set_yticks(range(4)); a.set_yticklabels([f"P{i}" for i in range(1, 5)])
    a.xaxis.tick_top(); a.xaxis.set_label_position("top"); a.tick_params(length=0, pad=10, labelsize=50)
    a.set_xlabel("attention to paragraph", fontsize=50, labelpad=12)
    a.set_ylabel("[Pk] in the line", fontsize=50, labelpad=10)
    fig.text(a.get_position().x0 + a.get_position().width / 2, 0.995, title, ha="center", va="top", fontsize=60, fontweight="bold", color=col)
    for s in a.spines.values(): s.set_visible(False)
cax = fig.add_axes([0.27, 0.06, 0.46, 0.04])
cb = ColorbarBase(cax, cmap=DIV, norm=nm, orientation="horizontal")
tk = [-60, -30, 0, 30, 60]; cb.set_ticks(tk); cb.set_ticklabels([MN(f"{t:+d}%") if t else "0%" for t in tk])
cb.outline.set_visible(False); cax.tick_params(labelsize=46, length=0, pad=8)
fig.text(0.26, 0.08, "less attention", ha="right", va="center", fontsize=48, fontweight="bold", color=RED)
fig.text(0.74, 0.08, "more attention", ha="left", va="center", fontsize=48, fontweight="bold", color=PURPLE)
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
<div class="ref">% = change in attention from the answer to each paragraph,<br>against the same prompt with the “Note that …” line;<br>± = the 95% confidence interval</div></div>
<img class="maps" src="data:image/png;base64,{maps}"></body></html>'''
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
hp = os.path.join(HERE, "rotate2.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "rotate2_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=2300,4000",
                "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, 2300, nz[-1] + 20)); os.remove(raw); os.remove(hp)
im.save(os.path.join(HERE, "rotate2.png")); im.save(os.path.join(HERE, "rotate2.jpg"), quality=92); print(im.size, round(im.size[1] / im.size[0], 2))
