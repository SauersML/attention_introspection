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
      ("8_do_not", "do not\npay\nattention"), ("9_completely_ignore", "completely\nignore")]
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
# violins on a log scale (log of each prompt's attention ratio, ticks labelled as % change, as in the heroes); the big number is the mean % change
from scipy.stats import gaussian_kde
L = lambda pc: np.log1p(np.asarray(pc) / 100)
FH_IN = 15.6; AH = FH_IN * 100 * 0.99
allz = np.concatenate([L(VM[c]["z"]) for c, _ in CS]); zlo, zhi = allz.min(), allz.max()
u = (zhi - zlo) / 900   # data units per px: the violins span about 900 px
yhi = zhi + 175 * u; ylo = yhi - AH * u
BB = ylo + 12 * u; BH = 225 * u   # band of wording boxes at the bottom (x-axis labels)
fig = plt.figure(figsize=(VW / 100, FH_IN))
a = fig.add_axes([0.004, 0.005, 0.994, 0.99])
rng = np.random.default_rng(1); PV = {}
for p, (c, lab) in enumerate(CS):
    zp = np.array(VM[c]["z"]); z = L(zp); m = VM[c]["mean"][0]; up = m > 0; oc = PURPLE if up else RED
    kde = gaussian_kde(z); yy = np.linspace(z.min() - 0.01, z.max() + 0.01, 300); dd = kde(yy); hw = 0.42 * dd / dd.max()
    a.fill_betweenx(yy, p - hw, p + hw, color=oc, alpha=0.16, lw=0, zorder=1)
    a.plot(np.r_[p + hw, p - hw[::-1], p + hw[:1]], np.r_[yy, yy[::-1], yy[:1]], color=oc, lw=2.5, zorder=2)
    xj = p + rng.uniform(-0.8, 0.8, len(z)) * np.interp(z, yy, hw)
    a.scatter(xj, z, s=230, c=pct_colour(1 + zp / 100), alpha=0.5, edgecolors="none", zorder=3); a.scatter(xj, z, s=4, color="black", zorder=4)
    a.plot([p - 0.13, p + 0.13], [L(m)] * 2, color=INK, lw=7, solid_capstyle="butt", zorder=6)   # mean mark
    pt = signflip(zp); PV[c] = pt
    if m > 5:   # % and p above the violin
        a.annotate(pt, (p, yy[-1]), xytext=(0, 4), textcoords="offset points", fontsize=44, color=COL[c], ha="center", va="bottom", zorder=7)
        a.annotate(MN(f"{m:+.0f}%"), (p, yy[-1]), xytext=(0, 50), textcoords="offset points", fontsize=64, fontweight="bold", color=COL[c], ha="center", va="bottom", zorder=7)
    else:       # below
        a.annotate(MN(f"{m:+.0f}%"), (p, yy[0]), xytext=(0, -4), textcoords="offset points", fontsize=64, fontweight="bold", color=COL[c], ha="center", va="top", zorder=7)
        a.annotate(pt, (p, yy[0]), xytext=(0, -70), textcoords="offset points", fontsize=44, color=COL[c], ha="center", va="top", zorder=7)
    # x-axis label: the boxed words of the prompt, one equal dashed box per wording (as in the bubble)
    a.text(p, BB + BH - 14 * u, lab, fontsize=37, fontweight="bold", color=COL[c], ha="center", va="top", linespacing=1.0, zorder=7)
X0 = -0.52   # y axis
a.plot([X0, len(CS) - 0.45], [0, 0], color=INK, lw=4, ls=(0, (4, 3)), zorder=5)
nm = VM["0_none"]["mean"][0]
a.plot([X0, len(CS) - 0.45], [L(nm)] * 2, color=INK, lw=2.5, ls=(0, (1.5, 2)), zorder=5)
a.text(X0 + 0.12, L(nm) - 6 * u, MN(f"no instruction line {nm:+.0f}%"), fontsize=40, fontweight="bold", ha="left", va="top", zorder=8)
a.plot([X0, X0], [L(-25) - 20 * u, L(100) + 20 * u], color=INK, lw=4, solid_capstyle="butt")
for t in (-25, 0, 25, 50, 100):
    a.plot([X0 - 0.05, X0], [L(t)] * 2, color=INK, lw=4)
    a.text(X0 - 0.08, L(t), MN(f"{t:+d}%") if t else "0%", fontsize=44, fontweight="bold", ha="right", va="center")
a.text(-1.9, L(37), "% change in attention\nto the named paragraph", rotation=90, fontsize=44, fontweight="bold", ha="left", va="center", linespacing=1.05, multialignment="center")
a.set_xlim(-1.92, len(CS) - 0.45); a.set_ylim(ylo, yhi)
a.set_yticks([]); a.set_xticks([])
for s_ in a.spines.values(): s_.set_visible(False)
# key, in the free space above the near-zero wordings: what a dot is and what 0% is
KX = 3.55; dotc = pct_colour([1.4]); KY = L(78)
a.scatter([KX + 0.08], [KY], s=900, c=dotc, alpha=0.6, edgecolors="none", zorder=8); a.scatter([KX + 0.08], [KY], s=10, color="black", zorder=9)
a.text(KX + 0.22, KY, "= one prompt", fontsize=44, fontweight="bold", ha="left", va="center", zorder=8)
a.text(KX, L(42), "0% =", fontsize=44, fontweight="bold", ha="left", va="center", zorder=8)
a.text(KX + 0.62, L(42), ment.replace("one of the ", "one of the\n").replace("is labelled", "\nis labelled"), fontsize=36, fontweight="bold", ha="left", va="center", linespacing=1.1, zorder=8,
       bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", fc="white", ec="#1F8BFF", lw=9))
LAD = os.path.join(TMP, "_dose_ladder.png"); fig.savefig(LAD, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- method on top + ladder, rendered with headless Chrome
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
DOSE = base64.b64encode(open(LAD, "rb").read()).decode()
def para(p):
    lab, txt = p.split("] ", 1); i = int(lab[2:]); txt = " ".join(txt.split()[:3]) + " …"
    return f'<div class="pl"><b class="{"k" if i == K else ""}">{html.escape(lab)}]</b> <span {"id=p2" if i == K else ""}>{html.escape(txt)}</span></div>'
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 28px;width:max-content;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.var{{background:#fff;color:{INK};border-radius:16px;padding:10px 16px;font-weight:800;font-size:54px;line-height:1.18}}
.dv{{border:5px dashed {INK};border-radius:12px;padding:0 8px}}
.vrow{{display:flex;align-items:center;gap:22px;margin:4px 0}} .lab{{font-size:50px;font-weight:800;color:#fff;white-space:nowrap}}
.mrow{{display:flex;align-items:center;gap:16px;margin-bottom:6px}}
.top{{display:flex;align-items:flex-end;gap:26px;position:relative}}
.mb#ans{{border:6px solid {PURPLE}}} .rd{{font-size:48px;font-weight:800;color:{PURPLE};line-height:1.1;margin-left:6px;white-space:nowrap}}
#ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}} .mrow{{position:relative;z-index:2}} .ub{{flex-shrink:0;position:relative;z-index:0}}
.rob{{width:110px;height:110px;flex-shrink:0}} .mcol{{display:flex;flex-direction:column;align-items:flex-start;gap:8px}}
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
<div class="mcol"><div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb" id="ans">{html.escape(ex["ans_free"].strip())}</div></div><div class="rd">attention read from<br>the answer tokens</div></div>
</div></div>
<img class="dose" src="data:image/png;base64,{DOSE}"><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
const sv=document.getElementById('ov'); sv.setAttribute('width',document.body.scrollWidth); sv.setAttribute('height',document.body.scrollHeight);
const a=R('ans'), t=R('p2');
const P0=[a.left+a.width/2, a.top+30], T=[t.right+30, (t.top+t.bottom)/2];
const C1=[P0[0], T[1]+40], C2=[T[0]+260, T[1]];
let ux=T[0]-C2[0], uy=T[1]-C2[1]; const n=Math.hypot(ux,uy); ux/=n; uy/=n; const Lh=58, Hh=36;
const E=[T[0]-Lh*ux, T[1]-Lh*uy];
const d=`M${{P0}} C${{C1}} ${{C2}} ${{E}}`, h=`M${{T}} L${{E[0]-Hh*uy}},${{E[1]+Hh*ux}} L${{E[0]+Hh*uy}},${{E[1]-Hh*ux}} Z`;
sv.innerHTML=`<path d="${{d}}" fill="none" stroke="{INK}" stroke-width="34" stroke-linecap="butt"/><path d="${{h}}" fill="#D9C8F5" stroke="{INK}" stroke-width="7" stroke-linejoin="round"/><path d="${{d}}" fill="none" stroke="#D9C8F5" stroke-width="22" stroke-linecap="butt"/>`;
</script></body></html>'''
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
hp = os.path.join(HERE, "dose_setup.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "dose_setup_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},5000",
                "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
os.remove(LAD)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, "dose_full.png")); im.save(os.path.join(HERE, "dose_full.jpg"), quality=92); print(im.size)
print({c: round(VM[c]["mean"][0], 1) for c, _ in CS}, "none", round(nm, 1)); print(PV)
