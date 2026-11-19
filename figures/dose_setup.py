"""dose_full.png: the wording ladder with its method. Top: the real prompt (id 300, "focus your attention on" wording, paragraphs
cut to three words) as a chat bubble, the instruction line in a white box with the words that change between wordings in a dashed
box; Qwen3-32B's reply; the 0% reference line (mention only) said beside the reply.
Below: the ladder, drawn here from dose_results.json (the same numbers dose_fig.py draws into dose.png): per prompt (150 arithmetic
prompts), attention from the answer tokens to the named paragraph, averaged over layers 21-53, as % change against the same prompt
whose line only mentions the paragraph (vs_mention[c]["z"]); one violin per wording, one dot per prompt coloured by its % change;
big number = mean over prompts (vs_mention[c]["mean"][0]); dashed lines at 0% (mention only) and at the no-instruction-line mean.
dose_results.json reports bootstrap CIs, not p-values, so only the mean is printed. Inset: each wording's per-layer mean
(per_layer_vs_mention) with the averaged band bracketed. Linear axis."""
import json, base64, subprocess, os, html, sys
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK, pct_colour
D = os.path.join(HERE, "..", "results", "qwen3-32b-dose") + "/"
meta = json.load(open(D + "meta.json")); R = json.load(open(D + "dose.json"))
ex = [r for r in R if r["id"] == 300 and r["cond"] == "3_focus"][0]
u = ex["user"]; K = ex["k"] + 1
line = meta["conds"]["3_focus"][0].format(k=K); assert line in u
ment = meta["conds"]["1_mention"][0].format(k=K)
VAR = "Focus your attention on"; assert line.startswith(VAR)
parts = u.split("\n\n"); paras = parts[1:5]; tail = "\n\n".join(parts[5:])
pre, post = tail.split(line)
AMB = "#F2B53A"; W = 2300; VW = W - 52
MN = lambda t: t.replace("-", "\u2212")
TMP = os.environ.get("FIGTMP", HERE)

# ---------------- the ladder (matplotlib), same data as dose_fig.py
RES = json.load(open(D + "dose_results.json")); VM = RES["vs_mention"]; PLY = RES["per_layer_vs_mention"]; B0, B1 = RES["layers"]
CS = [("2_glance", "glance\nat"), ("3_focus", "focus\nyour\nattention\non"), ("4_focus_strongly", "focus\nstrongly\non"),
      ("5_very_strongly_only", "focus\nvery\nstrongly\nand only"), ("7_little_less", "a little\nless"),
      ("8_do_not", "do not\npay\nattention"), ("9_completely_ignore", "com-\npletely\nignore")]
CS = sorted(CS, key=lambda c: -VM[c[0]]["mean"][0])
def tcol(m):
    """text/line colour for a wording, as in the heroes: purple for more attention, red for less; a change under 4% is INK"""
    if abs(np.log2(1 + m / 100)) < 0.06: return INK
    return PURPLE if m > 0 else RED
from pfmt import pstr
def signflip(z, NF=200000, seed=0):   # two-sided sign-flip test over prompts on the mean % change (as hero_L28.py)
    rng = np.random.default_rng(seed); obs = abs(z.mean()); hits = 0
    for _ in range(NF // 20000):
        s_ = rng.choice([-1.0, 1.0], (20000, len(z))); hits += int((np.abs((s_ * z).mean(1)) >= obs).sum())
    return pstr((hits + 1) / (NF + 1), floor=1 / (NF + 1))
COL = {c: tcol(VM[c]["mean"][0]) for c, _ in CS}
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK})
FH_IN = 16.6; BT, BH = 152, 30   # top band of wording boxes (data units, % change)
fig = plt.figure(figsize=(VW / 100, FH_IN))
a = fig.add_axes([0.004, 0.005, 0.994, 0.99])
rng = np.random.default_rng(1); PV = {}
for p, (c, lab) in enumerate(CS):
    z = np.array(VM[c]["z"]); m = VM[c]["mean"][0]; up = m > 0; oc = PURPLE if up else RED
    v = a.violinplot([z], positions=[p], widths=0.84, showextrema=False)
    for b in v["bodies"]: b.set_facecolor(oc); b.set_alpha(0.16); b.set_edgecolor(oc); b.set_linewidth(2.5)
    for b in v["bodies"]:   # crisp outline on top of the translucent fill
        b2 = plt.Polygon(b.get_paths()[0].vertices, closed=True, fill=False, ec=oc, lw=2.5, zorder=2); a.add_patch(b2)
    xj = p + rng.uniform(-0.27, 0.27, len(z))
    a.scatter(xj, z, s=230, c=pct_colour(1 + z / 100), alpha=0.5, edgecolors="none", zorder=3); a.scatter(xj, z, s=4, color="black", zorder=4)
    a.plot([p - 0.13, p + 0.13], [m, m], color=INK, lw=7, solid_capstyle="butt", zorder=6)   # mean mark
    lab_up = m > 5   # labels for the near-zero wordings sit under their dots, clear of the inset; p under the %, as in the heroes
    pt = signflip(z); PV[c] = pt
    if lab_up:
        a.text(p, z.max() + 2.0, pt, fontsize=44, color=COL[c], ha="center", va="bottom", zorder=7)
        a.text(p, z.max() + 8.5, MN(f"{m:+.0f}%"), fontsize=72, fontweight="bold", color=COL[c], ha="center", va="bottom", zorder=7)
    else:
        a.text(p, z.min() - 2.0, MN(f"{m:+.0f}%"), fontsize=72, fontweight="bold", color=COL[c], ha="center", va="top", zorder=7)
        a.text(p, z.min() - 15.0, pt, fontsize=44, color=COL[c], ha="center", va="top", zorder=7)
    # the boxed words of the prompt, one equal box per wording (dashed, as in the bubble), in a band at the top: wording above its result
    a.text(p, BT - 1.2, lab, fontsize=38, fontweight="bold", color=COL[c], ha="center", va="top", linespacing=1.0, zorder=7, clip_on=False)
    a.add_patch(FancyBboxPatch((p - 0.45, BT - BH), 0.90, BH, boxstyle="round,pad=0,rounding_size=0.12", mutation_aspect=BH / 0.90 * 0.22,
                               fc="white", ec=INK, lw=3.5, ls=(0, (4, 3)), zorder=6, clip_on=False))
a.axhline(0, color=INK, lw=4, ls=(0, (4, 3)), zorder=5)
nm = VM["0_none"]["mean"][0]
a.axhline(nm, color=INK, lw=2.5, ls=(0, (1.5, 2)), zorder=5)
a.text(-1.03, 1.4, "mention\nonly\n= 0%", fontsize=40, fontweight="bold", ha="left", va="bottom", linespacing=1.0, zorder=8)
a.text(-1.03, nm - 1.4, MN(f"no instruction line {nm:+.0f}%"), fontsize=40, fontweight="bold", ha="left", va="top", zorder=8)
a.set_xlim(-1.05, len(CS) - 0.43); a.set_ylim(-52, BT + 1)
a.set_yticks([]); a.set_xticks([])   # no y axis (as the hero): the dashed 0% line and the big means carry the scale
for s in a.spines.values(): s.set_visible(False)
# key, in the free space above the near-zero wordings: what a dot is and what 0% is
KX = 3.58; dotc = pct_colour([1.4])
a.scatter([KX + 0.08], [103], s=900, c=dotc, alpha=0.6, edgecolors="none", zorder=8); a.scatter([KX + 0.08], [103], s=10, color="black", zorder=9)
a.text(KX + 0.22, 103, "= one prompt", fontsize=44, fontweight="bold", ha="left", va="center", zorder=8)
a.text(KX, 90, "% change in attention to the\nnamed paragraph, compared to:", fontsize=38, fontweight="bold", ha="left", va="top", linespacing=1.1, zorder=8)
a.text(KX + 0.06, 62, ment.replace("paragraphs ", "paragraphs\n"), fontsize=36, fontweight="bold", ha="left", va="top", linespacing=1.1, zorder=8,
       bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", fc="white", ec="#1F8BFF", lw=9))
LAD = os.path.join(TMP, "_dose_ladder.png"); fig.savefig(LAD, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- method on top + ladder, rendered with headless Chrome
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
DOSE = base64.b64encode(open(LAD, "rb").read()).decode()
def para(p):
    lab, txt = p.split("] ", 1); i = int(lab[2:]); txt = " ".join(txt.split()[:3]) + " …"
    return f'<div class="pl"><b class="{"k" if i == K else ""}">{html.escape(lab)}]</b> {html.escape(txt)}</div>'
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 28px;width:max-content;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.var{{background:#fff;color:{INK};border-radius:16px;padding:10px 16px;font-weight:800;font-size:54px;line-height:1.18}}
.dv{{border:5px dashed {INK};border-radius:12px;padding:0 8px}}
.vrow{{display:flex;align-items:center;gap:22px;margin:4px 0}} .lab{{font-size:50px;font-weight:800;color:#fff;white-space:nowrap}}
.mrow{{display:flex;align-items:center;gap:16px;margin-bottom:6px}}
.top{{display:flex;align-items:flex-end;gap:26px}}
.rob{{width:110px;height:110px;flex-shrink:0}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:70px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.ref{{margin-left:34px;display:flex;flex-direction:column;gap:10px}}
.r1{{font-size:50px;font-weight:800;display:flex;align-items:center;gap:14px;white-space:nowrap}}
.dot{{display:inline-block;width:46px;height:46px;border-radius:50%;background:#7A55C9;opacity:.6;position:relative}}
.dot:after{{content:"";position:absolute;left:20px;top:20px;width:6px;height:6px;border-radius:50%;background:#000}}
.cub{{background:#1F8BFF;border-radius:24px;padding:10px 12px;align-self:flex-start}}
.cub .var{{font-size:46px;padding:6px 14px}}
.dose{{width:100%;display:block;margin-top:18px}}
</style></head><body>
<div class="top"><div class="ub"><div>{html.escape(parts[0].strip())}</div>{"".join(para(p) for p in paras)}
<div class="vrow"><div class="var"><span class="dv">{html.escape(VAR)}</span>{html.escape(line[len(VAR):])}</div></div><div>{html.escape(post.strip())}</div></div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(ex["ans_free"].strip())}</div>
</div></div>
<img class="dose" src="data:image/png;base64,{DOSE}"></body></html>'''
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
hp = os.path.join(HERE, "dose_setup.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "dose_setup_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},5000",
                "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
os.remove(LAD)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, "dose_full.png")); im.save(os.path.join(HERE, "dose_full.jpg"), quality=92); print(im.size)
print({c: round(VM[c]["mean"][0], 1) for c, _ in CS}, "none", round(nm, 1)); print(PV)
