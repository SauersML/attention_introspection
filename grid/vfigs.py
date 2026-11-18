# Three violin figures from the "paragraph"-wording grid run (Qwen3-32B, arithmetic, 100 contexts x 4 named paragraphs),
# in the style of vhero.py + hero_setup.py. One point = one (context, named paragraph) pair; mean and 95% bootstrap CI
# resample contexts (each with its 4 named paragraphs). Effect = % change of attention on the named paragraph.
import json, sys, os, html, base64, subprocess, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
HERE = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
DATA = "/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/grid/"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")); from pal import PURPLE, RED, INK, pct_colour
R = json.load(open(sys.argv[1])); IDS = [r["id"] for r in R]
MINUS = "\u2212"
LINE = {"focus": "Focus your attention on paragraph [P{k}].", "ignore": "Do not pay attention to paragraph [P{k}].",
        "mention": "Note that one of the paragraphs is labelled [P{k}].", "none": ""}
NAME = {"focus": "focus your attention", "ignore": "do not pay attention", "mention": "only mentioned"}
REF = {"none": "no instruction", "mention": "only mentioned"}
def att(r, c, k): return r["vers"]["none" if c == "none" else f"{c}{k}"]["span_band"][k - 1]
def pairs(c, ref):
    return np.array([[100 * (att(r, c, k) / att(r, ref, k) - 1) for k in range(1, 5)] for r in R])  # [ctx, 4]
def boot(Z, NB=10000, seed=0):
    rng = np.random.default_rng(seed); i = rng.integers(0, len(Z), (NB, len(Z))); s = Z[i].reshape(NB, -1).mean(1)
    return Z.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)
def signflip(Z, NB=10000, seed=1):  # per-context mean, random sign per context
    m = Z.mean(1); rng = np.random.default_rng(seed); s = rng.choice([-1, 1], (NB, len(m))); null = (s * m).mean(1)
    return (np.sum(np.abs(null) >= abs(m.mean())) + 1) / (NB + 1)

# example context for the bubbles: the first context, named paragraph P3 (as in the hero)
EX = R[0]; K = 3
assert all(v["free_same"] for v in EX["vers"].values())
u = EX["vers"]["focus3"]["user"]; body = u.split("\n\n"); paras = [p.split("] ", 1)[1] for p in body[1:5]]
Q = body[5].split("\n")[-1]; ANS = EX["ans_none"]
for c in ("focus", "ignore", "mention"): assert LINE[c].format(k=K) in EX["vers"][f"{c}{K}"]["user"]
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
COL = {"focus": PURPLE, "ignore": RED, "mention": PURPLE}

def violins(res, ref, png, W):
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 56, "text.color": INK})
    n = len(res); fig, a = plt.subplots(figsize=(W / 100, 8.2))
    allz = np.concatenate([r["z"].ravel() for r in res])
    rng = np.random.default_rng(1); GS = max(allz.max(), 0) - min(allz.min(), 0)
    for p, r in enumerate(res):
        z = r["z"].ravel(); x0 = p * 1.35; col = PURPLE if r["mean"] > 0 else RED
        vv = a.violinplot([z], positions=[x0], widths=0.7, showextrema=False)
        for bb in vv["bodies"]: bb.set_facecolor(col); bb.set_alpha(0.22); bb.set_edgecolor(col); bb.set_linewidth(2.5)
        xj = x0 + rng.uniform(-0.25, 0.25, len(z)); a.scatter(xj, z, s=650, c=pct_colour(1 + z / 100), alpha=0.45, edgecolors="none", zorder=3); a.scatter(xj, z, s=6, color="black", zorder=4)
        a.plot([x0, x0], [r["lo"], r["hi"]], color=INK, lw=12, solid_capstyle="butt", zorder=5); a.plot([x0 - 0.22, x0 + 0.22], [r["mean"]] * 2, color=INK, lw=6, zorder=6)
        a.text(x0, (z.max() + 0.03 * GS) if r["mean"] > 0 else (z.min() - 0.05 * GS), f"{r['mean']:+.0f}% attention".replace("-", MINUS), fontsize=100, fontweight="bold",
               color=col, ha="center", va=("bottom" if r["mean"] > 0 else "top"), zorder=7)
    a.axhline(0, color=INK, lw=4, zorder=4)
    lo = min(allz.min(), 0); hi = max(allz.max(), 0); span = hi - lo
    neg = [r for r in res if r["mean"] < 0]
    a.set_ylim(lo - (0.42 * span if neg else 0.06 * span) - 0.13 * span, hi + (0.38 * span if max(r["mean"] for r in res) > 0 else 0.12 * span))
    x_left = -0.66 if n > 1 else -1.5  # one violin: centred, with room on its left for the baseline label
    a.set_xlim(x_left, (n - 1) * 1.35 + (0.66 if n > 1 else 1.5))
    a.text(x_left + 0.04, -0.02 * span, f"0 = {REF[ref]}" if n > 1 else f"0 = {REF[ref].replace(' ', chr(10), 1)}", fontsize=60, fontweight="bold", color=INK, ha="left", va="top", zorder=8)
    for s_ in a.spines.values(): s_.set_visible(False)
    a.set_xticks([]); a.set_yticks([]); fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(png, dpi=100, facecolor="white"); plt.close(fig)

def column(c):
    pl = "".join(f'<div class="pl"><b class="{"k" if i + 1 == K else ""}">[P{i + 1}]</b> {html.escape(" ".join(p.split()[:2 if i == 0 else 3]))} …</div>' for i, p in enumerate(paras))
    return f'''<div class="col"><div class="cname" style="color:{COL[c]}">{NAME[c]}</div><div class="ub"><div>Here are four paragraphs.</div>{pl}
<div class="var" style="color:{COL[c]}">{html.escape(LINE[c].format(k=K))}</div><div>{html.escape(Q).replace(" + ", "&nbsp;+&nbsp;")}</div></div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(ANS)}</div></div></div>'''

def make(conds, ref, name, W):
    res = []
    for c in conds:
        Z = pairs(c, ref); m, lo, hi = boot(Z); p = signflip(Z)
        res.append(dict(cond=c, z=Z, mean=m, lo=lo, hi=hi, p=p))
        print(f"{name}: {c} vs {ref}: mean {m:+.2f}% [{lo:+.2f}, {hi:+.2f}], {Z.size} pairs, {len(Z)} contexts, sign-flip p {p:.1e}, pairs above 0 {(Z > 0).mean() * 100:.0f}%")
    vpng = HERE + name + "_vio.png"; violins(res, ref, vpng, W - 52)
    vio = base64.b64encode(open(vpng, "rb").read()).decode()
    page = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.cols{{display:flex;gap:36px}} .col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:76px;font-weight:800;margin:0 0 12px 4px;white-space:nowrap}}
.ub{{flex:1;background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 26px;font-size:54px;line-height:1.18;display:flex;flex-direction:column;gap:8px}}
.pl{{white-space:nowrap;overflow:hidden}} .var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18}}
.k{{background:#F2B53A;color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:12px}} .rob{{width:110px;height:110px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:70px;font-weight:700}}
.vio{{width:100%;display:block}}
</style></head><body><div class="cols">{"".join(column(c) for c in conds)}</div><img class="vio" src="data:image/png;base64,{vio}"></body></html>'''
    open(HERE + name + ".html", "w").write(page)
    CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    raw = HERE + name + "_raw.png"
    subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3200",
                    "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + HERE + name + ".html"], check=True, capture_output=True)
    im = Image.open(raw).convert("RGB"); a = np.asarray(im); nz = np.where((a < 250).any(2).any(1))[0]
    im = im.crop((0, 0, W, nz[-1] + 20)); im.save(HERE + name + ".png"); im.save(HERE + name + ".jpg", quality=92); print(name, im.size)
    return [dict(cond=r["cond"], ref=ref, mean_pct=r["mean"], ci95=[r["lo"], r["hi"]], signflip_p=r["p"], n_pairs=int(r["z"].size), n_contexts=len(r["z"]),
                 per_pair_pct={str(i): r["z"][j].tolist() for j, i in enumerate(IDS)}) for r in res]

OUTJ = dict(model="Qwen/Qwen3-32B", task="arith", wording=LINE, layers="21-53 of 64", unit="one point = one (context, named paragraph k=1..4) pair; per_pair_pct lists k=1..4",
            ci="95% bootstrap over contexts (10,000)", p="two-sided sign-flip over per-context means (10,000)")
OUTJ["v_none"] = make(["focus", "ignore"], "none", "v_none", 2300)
OUTJ["v_mention"] = make(["focus", "ignore"], "mention", "v_mention", 2300)
OUTJ["v_mention_vs_none"] = make(["mention"], "none", "v_mention_vs_none", 1700)
OUTJ["free_answers_same"] = [sum(v["free_same"] for r in R for kk, v in r["vers"].items() if kk != "none"), sum(1 for r in R for kk in r["vers"] if kk != "none")]
json.dump(OUTJ, open(DATA + "violins.json", "w"), indent=1)
print("free answers same", OUTJ["free_answers_same"])
