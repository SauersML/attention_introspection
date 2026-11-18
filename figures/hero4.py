"""Four-column hero (Qwen3-32B): two controls (instructions in the prompt) and two injections (attention boosted onto paragraph 2).
  focus     results/qwen3-32b-dose, layer 28: per context, % change in attention from the answer tokens to the named paragraph
            (head mean, per-token mean over the paragraph, as hero_L28.py) under "Focus your attention on paragraph [Pk]." vs the
            mention-only line "Note that one of the paragraphs is labelled [Pk]."; 150 contexts. Violin on log2 of the ratio.
  ignore    same, "Completely ignore paragraph [Pk]. Do not look at it at all."
  question  results/privileged_inorder/data (labels in order), arm C (boost while the report question is read), multiplier 10, wording W1,
            100 paragraph prompts. Per prompt: % change vs no injection in how often the top (argmax) answer is the boosted paragraph:
            its rate over the 4 rotations / 25% - 1 (no-injection rate is exactly 25% by rotation). Example: as hero_top.py question mode.
  told      results/told_inject/data, multiplier 10, the same metric on the top paragraph after "Yes, paragraph [P"; 240 prompts. Example: context 101.
Layout: the two controls share one chat (identical except the instruction line, drawn as two boxes side by side, same answer 163);
the injection experiments differ in most lines (told has the "I can boost..." intro and the "Do you notice...?" question, question has the
arithmetic answer and the report question), so each gets its own chat. Two y axes: one per pair (different measures).
mean   (default): black line = mean, big label = mean, p = two-sided sign-flip test (200,000 flips)           -> hero4.png
median          : black line = median, big label = median, p = exact two-sided sign test with zero differences split evenly between
                  the two signs (tests median = 0; the zero-dropping sign test is printed too)                  -> hero4_median.png
95% bootstrap CIs (10,000 resamples over prompts) are printed, not drawn.
usage: nice -n 19 python3 hero4.py [mean|median]"""
import json, glob, os, sys, html, base64, subprocess
import numpy as np, matplotlib
import matplotlib.patheffects as pe
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, binomtest, wilcoxon
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from robot_injected import injected_robot
STAT = sys.argv[1] if len(sys.argv) > 1 else "mean"; assert STAT in ("mean", "median")
BLUE, RED, VIO, LIL, AMB = "#2B5FA8", "#C0392B", "#7B3FE4", "#A574F2", "#F2B53A"
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
neg = lambda s: s.replace("-", "\u2212")
short = lambda s, n: " ".join(s.split()[:n]) + " \u2026"
recs = lambda d: sorted([json.loads(l) for f in glob.glob(d + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])

# ---------------- data ----------------
D = os.path.join(ROOT, "results/qwen3-32b-dose/"); meta = json.load(open(D + "meta.json"))
RD = json.load(open(D + "dose.json")); by = {(r["id"], r["cond"]): r for r in RD}; cids = sorted({r["id"] for r in RD})
att = lambda r: np.array(r["att_mean"])[28][r["k"]]
A = {c: np.array([att(by[i, c]) for i in cids]) for c in ("1_mention", "3_focus", "9_completely_ignore")}
def top_pc(cond, none, n):   # % change vs no injection in the rate of top answer = boosted paragraph (no-injection rate is 1/n by rotation)
    hit = np.mean([int(np.argmax(cond(k))) == k for k in range(n)]); base = np.mean([int(np.argmax(none)) == k for k in range(n)])
    assert abs(base - 1 / n) < 1e-9; return 100 * (hit / base - 1)
RQ = [r for r in recs(os.path.join(ROOT, "results/privileged_inorder/data/")) if r["setup"] in ("par_arith", "par_qa")]; assert len(RQ) == 100
RT = recs(os.path.join(ROOT, "results/told_inject/data/")); assert len(RT) == 240
Z = {"focus": 100 * (A["3_focus"] / A["1_mention"] - 1), "ignore": 100 * (A["9_completely_ignore"] / A["1_mention"] - 1),
     "question": np.array([top_pc(lambda k, r=r: r["C"][f"{k}|10.0"]["W1"], r["none"]["W1"], r["N"]) for r in RQ]),
     "told": np.array([top_pc(lambda k, r=r: r["boost"][f"{k}|10.0"]["lp_par"], r["none"]["lp_par"], 4) for r in RT])}

def signflip(x, NF=200000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(x.mean()); h = 0
    for _ in range(NF // 20000):
        s = rng.choice([-1.0, 1.0], (20000, len(x))); h += int((np.abs((s * x).mean(1)) >= obs).sum())
    return (h + 1) / (NF + 1), NF
def signtest_split(x):
    pos, ng, z0 = int((x > 0).sum()), int((x < 0).sum()), int((x == 0).sum())
    n = pos + ng + 2 * (z0 // 2); return binomtest(pos + z0 // 2, n).pvalue
ST = {}
for k, z in Z.items():
    idx = np.random.default_rng(0).integers(0, len(z), (10000, len(z)))
    pf, nf = signflip(z); pos, ng = int((z > 0).sum()), int((z < 0).sum())
    ST[k] = dict(n=len(z), mean=z.mean(), mean_ci=np.percentile(z[idx].mean(1), [2.5, 97.5]), p_flip=pf, nf=nf,
                 median=np.median(z), med_ci=np.percentile(np.median(z[idx], 1), [2.5, 97.5]), p_sign_split=signtest_split(z),
                 p_sign_drop=binomtest(pos, pos + ng).pvalue, p_wilcoxon=wilcoxon(z).pvalue, pos=pos, neg=ng, zero=int((z == 0).sum()))
    s = ST[k]
    print(f"{k:8s} n={s['n']}  mean {s['mean']:+.1f}% [{s['mean_ci'][0]:+.1f}, {s['mean_ci'][1]:+.1f}] p_signflip={s['p_flip']:.2g}   "
          f"median {s['median']:+.1f}% [{s['med_ci'][0]:+.1f}, {s['med_ci'][1]:+.1f}] p_sign(ties split)={s['p_sign_split']:.2g} "
          f"p_sign(ties dropped)={s['p_sign_drop']:.2g} p_wilcoxon={s['p_wilcoxon']:.2g}  +{pos} -{ng} 0:{s['zero']}")
def ptxt(k):
    s = ST[k]
    if STAT == "mean":
        if s["p_flip"] * (s["nf"] + 1) < 1.5: return "p < 0.00001"
        p = s["p_flip"]
    else: p = s["p_sign_split"]
    return "p < 0.00001" if p < 1e-5 else f"p = " + np.format_float_positional(p, precision=1, fractional=False) if p < 0.001 else f"p = {p:.2f}" if p < 0.1 else f"p = {p:.1f}"
cen = lambda z: float(np.mean(z) if STAT == "mean" else np.median(z))

# ---------------- examples ----------------
EX = 300; exf, exi = by[EX, "3_focus"], by[EX, "9_completely_ignore"]
assert exf["ans_free"].strip() == exi["ans_free"].strip(); ANS_C = exf["ans_free"].strip(); KC = exf["k"] + 1
uc = exf["user"]; paras_c = [p.split("] ", 1)[1] for p in uc.split("\n\n")[1:5]]; qline_c = uc.split("\n\n")[5].split("\n")[1]
LINE = {c: meta["conds"][c][0].format(k=KC) for c in ("3_focus", "9_completely_ignore")}
assert LINE["3_focus"] in exf["user"] and LINE["9_completely_ignore"] in exi["user"] and qline_c in exi["user"]
# question example: as hero_top.py (top answer with [P2] boosted is [P2], dot nearest the median; ties: labels in order, then lowest id)
zq = Z["question"]; medq = np.median(zq)
cand = [(abs(zq[i] - medq), r["lab"] != sorted(r["lab"]), r["id"]) for i, r in enumerate(RQ) if int(np.argmax(r["C"][f"{r['lab'].index(2)}|10.0"]["W1"])) == r["lab"].index(2)]
QID = min(cand)[2]; rq = {r["id"]: r for r in RQ}[QID]; KQ = rq["lab"].index(2)
EXQ = {c["id"]: c for c in json.load(open(os.path.join(HERE, "privileged_inorder_ctx.json")))}[QID]
assert EXQ["lab"] == rq["lab"] and EXQ["ntok"] == rq["ntok"]
QREPLY = f"P{rq['lab'][int(np.argmax(rq['C'][f'{KQ}|10.0']['W1']))]}"; QLINE = EXQ["user"].split("\n\n")[-1]
DOSEQ = np.mean([np.mean([r["massC"][f"{k}|10.0"][k] / r["mass_none_rep"][k] for k in range(r["N"])]) for r in RQ])
TID, KT = 101, 1; rt = {r["id"]: r for r in RT}[TID]; segs_t = {c["id"]: c for c in json.load(open(os.path.join(HERE, "told_contexts.json")))}[TID]["segs"]
TREPLY = f'Yes, paragraph [P{int(np.argmax(rt["boost"][f"{KT}|10.0"]["lp_par"])) + 1}'
DOSET = np.mean([np.mean([r["boost"][f"{k}|10.0"]["mass"][k] / r["none"]["mass"][k] for k in range(4)]) for r in RT])
print(f"examples: control id {EX} answer {ANS_C}; question id {QID} reply {QREPLY} dose x{DOSEQ:.1f}; told id {TID} reply {TREPLY!r} dose x{DOSET:.1f}")

# ---------------- layout (CSS px; page rendered at 2x) ----------------
W = 3000
CC = {"focus": 520, "ignore": 1160, "question": 1975, "told": 2650}          # violin centres
AXX = {"control": 192, "inject": 1672}                                         # y axes
PAIR = {"control": (20, 1470), "inject": (1530, 2980)}
COLR = {"focus": BLUE, "ignore": RED, "question": VIO, "told": VIO}
TXT = {"focus": BLUE, "ignore": RED, "question": PURPLE, "told": PURPLE}

# ---------------- result row (matplotlib, pixel coordinates) ----------------
VH = 1150; PT = lambda v: v * 72 / 100   # css px -> points at dpi 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(VH, 0); ov.axis("off")
PY0, PY1 = 440, VH - 30                       # plot band (px from top)
def ymap(lo, hi): return lambda v: PY1 - (PY1 - PY0) * (v - lo) / (hi - lo)
LC = np.log2(1 + np.concatenate([Z["focus"], Z["ignore"]]) / 100); YC = ymap(LC.min() - 0.2, LC.max() + 0.2)
YI = ymap(-100 - 56, 300 + 56)
def violin(k, Y, vals, jit, bw, halfw):
    z = vals; kde = gaussian_kde(z, bw_method=bw / z.std(ddof=1))
    yy = np.linspace(z.min() - 1.5 * bw, z.max() + 1.5 * bw, 500); dd = kde(yy); dd = dd / dd.max() * halfw; col = COLR[k]
    ov.fill_betweenx([Y(v) for v in yy], CC[k] - dd, CC[k] + dd, color=col, alpha=0.13, lw=0)
    ov.plot(np.r_[CC[k] - dd, (CC[k] + dd)[::-1]], [Y(v) for v in np.r_[yy, yy[::-1]]], color=col, lw=3, alpha=0.8)
    rng = np.random.default_rng(1); zj = z + rng.uniform(-jit, jit, len(z))
    xj = CC[k] + rng.uniform(-1, 1, len(z)) * kde(zj) / kde(yy).max() * halfw * 0.9
    ov.scatter(xj, [Y(v) for v in zj], s=PT(30) ** 2 * 0.8, color=col, alpha=0.38, edgecolors="none", zorder=3)
for k in ("focus", "ignore"):
    violin(k, YC, np.log2(1 + Z[k] / 100), 0, 0.12, 230)
for k in ("question", "told"):
    violin(k, YI, Z[k], 16, 36, 205)
# zero lines, axes, ticks
for pr, Y, ticks, lab in [("control", YC, [(-1, "\u221250%"), (0, "0"), (1, "+100%"), (2, "+300%"), (3, "+700%")], "only mentioning paragraph 2"),
                          ("inject", YI, [(t, neg(f"{t:+d}%") if t else "0") for t in (-100, 0, 100, 200, 300)], "no injection")]:
    x0, x1 = PAIR[pr]; ax_ = AXX[pr]
    ov.plot([ax_, x1], [Y(0)] * 2, color=INK, lw=4, zorder=1)
    ov.plot([ax_] * 2, [Y(ticks[0][0]), Y(ticks[-1][0])], color=INK, lw=3)
    for v, t in ticks:
        ov.plot([ax_ - 12, ax_], [Y(v)] * 2, color=INK, lw=3)
        ov.text(ax_ - 20, Y(v), t, ha="right", va="center", fontsize=PT(46), color=INK)
    if pr == "control": ov.text(ax_ + 24, Y(0) + 12, lab, ha="left", va="top", fontsize=PT(46), color=INK)
    else: ov.text((CC["question"] + CC["told"]) / 2, Y(0) - 10, lab.replace(" ", "\n"), ha="center", va="bottom", fontsize=PT(46), color=INK, linespacing=1.05)
# black centre lines, their label once per pair
for k, Y, tr in [("focus", YC, lambda v: np.log2(1 + v / 100)), ("ignore", YC, lambda v: np.log2(1 + v / 100)),
                 ("question", YI, lambda v: v), ("told", YI, lambda v: v)]:
    y = Y(tr(cen(Z[k]))); ov.plot([CC[k] - 150, CC[k] + 150], [y, y], color=INK, lw=6, zorder=6, solid_capstyle="butt")
    if k in ("focus", "question"):
        ov.text(CC[k] - 150, y - 8, STAT, ha="left", va="bottom", fontsize=PT(46), fontweight="bold", color=INK, zorder=6,
                path_effects=[pe.withStroke(linewidth=10, foreground="white")])
# axis titles (one per pair) and big numbers
YT = {"control": "change vs only mentioning paragraph 2\nin attention from the answer to paragraph 2",
      "inject": "change vs no injection in how often\nQwen\u2019s answer is the boosted paragraph"}
for pr in YT:
    ov.text((AXX[pr] - 140 + PAIR[pr][1]) / 2, 8, YT[pr], ha="center", va="top", fontsize=PT(54), fontweight="bold", color=INK, linespacing=1.12)
for k in CC:
    v = cen(Z[k]); s = neg(f"{v:+.0f}%") if round(v) else "0%"
    ov.text(CC[k], 300, s, ha="center", va="baseline", fontsize=PT(130), fontweight="bold", color=TXT[k])
    ov.text(CC[k], 318, ptxt(k), ha="center", va="top", fontsize=PT(52), color=TXT[k])
# key, inside the empty region below zero in the focus column
ky = YC(-0.78); ov.scatter([CC["focus"] - 150], [ky], s=PT(30) ** 2 * 0.8, color=BLUE, alpha=0.38, edgecolors="none")
ov.text(CC["focus"] - 118, ky, "= one prompt", ha="left", va="center", fontsize=PT(46), color=INK)
VP = os.path.join(HERE, f"_v4_{STAT}.png"); fig.savefig(VP, dpi=200, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
def syringe(i, dose):
    o = [f'<path d="M380,80 L364,76 L364,84 Z" fill="{INK}"/>', f'<rect x="314" y="76" width="50" height="8" fill="{INK}"/>',
         f'<path d="M314,68 L292,58 L292,102 L314,92 Z" fill="{INK}"/>',
         f'<rect x="80" y="22" width="212" height="116" rx="14" fill="#fff" stroke="{INK}" stroke-width="7"/>',
         f'<rect x="109" y="29" width="176" height="102" fill="{LIL}"/>', f'<rect x="87" y="29" width="22" height="102" fill="{INK}"/>',
         f'<rect x="62" y="4" width="18" height="152" rx="7" fill="{INK}"/>', f'<rect x="26" y="71" width="36" height="18" fill="{INK}"/>',
         f'<rect x="2" y="40" width="24" height="80" rx="7" fill="{INK}"/>',
         f'<text x="197" y="108" font-size="82" font-weight="800" text-anchor="middle" fill="#fff">\u00d7{dose:.1f}</text>']
    return f'<svg id="syr{i}" class="syr" width="380" height="160" viewBox="0 0 380 160">{"".join(o)}</svg>'
b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
robot = b64("/Users/user/watching/figures/assets/robot.png"); robot_inj = injected_robot("/Users/user/watching/figures/assets/robot.png")
def plines(segs, lab, k, pid, nw):
    return "".join(f'<div class="pl"><b {f"id={pid}c" if pos == k else ""}>[P{lab[pos]}]</b> '
                   f'<span {f"class=k id={pid}t" if pos == k else ""}>{html.escape(short(segs[pos], nw))}</span></div>' for pos in range(4))
plc = "".join(f'<div class="pl"><b>[P{i + 1}]</b> <span {"class=k" if i + 1 == KC else ""}>{html.escape(short(p, 3))}</span></div>' for i, p in enumerate(paras_c))
def arcs_svg(w=1450, h=300):
    """what the control measures: attention from the answer tokens to paragraph 2's tokens (a drawing of the measure, not values)"""
    o = []; sq, gp, gg, x, ty = 22, 5, 40, 40, 190; xs = []
    for g in range(4):
        gx = []
        for t in range(8):
            o.append(f'<rect x="{x}" y="{ty}" width="{sq}" height="{sq + 10}" rx="4" fill="{AMB if g == 1 else "#C9CCD3"}"/>'); gx.append(x + sq / 2); x += sq + gp
        if g == 1: o.append(f'<text x="{(gx[0] + gx[-1]) / 2:.0f}" y="{ty + sq + 70}" font-size="46" font-weight="700" fill="{INK}" text-anchor="middle">paragraph 2</text>')
        xs.append(gx); x += gg
    rx_ = x + 30; o.append(f'<image href="data:image/png;base64,{robot}" x="{rx_}" y="{ty + 18 - 60}" width="120" height="120"/>')
    bx = rx_ + 136; bw, bh = 150, 84; by_ = ty + 18 - bh / 2
    o.append(f'<rect x="{bx}" y="{by_}" width="{bw}" height="{bh}" rx="28" fill="#fff" stroke="#D9D9DE" stroke-width="3"/>')
    o.append(f'<text x="{bx + bw / 2}" y="{by_ + bh / 2 + 20}" font-size="58" font-weight="700" fill="{INK}" text-anchor="middle">{html.escape(ANS_C)}</text>')
    sx, sy = bx + bw / 2, by_ - 4
    for tx in xs[1]:
        o.append(f'<path d="M{sx:.0f},{sy:.0f} C{sx - 60:.0f},{ty - 150} {tx + 40:.0f},{ty - 150} {tx:.0f},{ty - 6}" fill="none" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')
    o.append(f'<text x="{(xs[1][-1] + sx) / 2 + 40:.0f}" y="{ty - 140}" font-size="46" font-weight="700" fill="{INK}" text-anchor="middle">attention at layer 28</text>')
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif">{"".join(o)}</svg>'

CTRL = f'''<div class="ub" style="font-size:58px">Here are four paragraphs.{plc}
  <div class="vrow"><div class="var" style="color:{BLUE}">{html.escape(LINE["3_focus"])}</div><div class="var" style="color:{RED}">{html.escape(LINE["9_completely_ignore"])}</div></div>
  <div>{html.escape(qline_c)}</div></div>
  <div style="margin-top:auto">{arcs_svg()}</div>'''
QCOL = f'''<div class="ub i">Here are four paragraphs.{plines(EXQ["segs"], EXQ["lab"], KQ, "q", 2)}<div>{html.escape(QLINE)}</div></div>
  <div class="mrow" style="margin-left:40px"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(rq["ans"])}</div></div>
  <div class="ub2" id="qq">Which paragraph did you attend to most while writing your answer? \u2026</div>
  <div class="mrow inj" style="margin-left:150px"><img class="rob" id="qr" src="data:image/png;base64,{robot_inj}"><div class="mb">{html.escape(QREPLY)}</div></div>'''
TCOL = f'''<div class="ub i">I can boost your attention onto one of these paragraphs, on half the trials.{plines(segs_t, [1, 2, 3, 4], KT, "t", 2)}</div>
  <div class="ub2" id="tq">Do you notice your attention being boosted onto one of the paragraphs? If so, which one?</div>
  <div class="mrow inj" style="margin-left:56px"><img class="rob" style="width:100px;height:100px" id="tr" src="data:image/png;base64,{robot_inj}"><div class="mb">{html.escape(TREPLY)}</div></div>'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};position:relative;padding-top:18px}}
.hd{{position:absolute;top:18px;font-size:66px;font-weight:800;padding-bottom:8px;border-bottom:7px solid}}
.top{{position:relative;height:var(--th)}}
.blk{{position:absolute;top:110px;display:flex;flex-direction:column}}
.ct{{font-size:56px;font-weight:800;color:{PURPLE};margin:0 0 12px 6px;white-space:nowrap}}
.ub{{background:#1F8BFF;color:#fff;border-radius:32px;padding:18px 26px;font-size:50px;line-height:1.2}}
.pl{{white-space:nowrap;overflow:hidden}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.vrow{{display:flex;gap:22px;margin:8px 0}}
.var{{flex:1;background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.15}}
.blk .ub.i{{margin-right:62px}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:32px;padding:16px 26px;font-size:50px;line-height:1.2;margin:56px 60px 0 150px}}
.mrow{{display:flex;align-items:center;gap:14px;margin-top:14px}}
.mrow.inj{{margin-top:auto;padding-top:40px}}
.rob{{width:112px;height:112px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:28px;padding:6px 22px;font-size:50px;font-weight:700;white-space:nowrap}}
.syr{{position:absolute;transform-origin:380px 80px}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}} #ov{{position:absolute;left:0;top:0;pointer-events:none}}
.vio{{width:{W}px;display:block}}
</style></head><body>
<div class="top" id="top">
 <div class="hd" style="left:{PAIR['control'][0]}px;width:{PAIR['control'][1]-PAIR['control'][0]}px;border-color:{INK}">control: instructions in the prompt</div>
 <div class="hd" style="left:{PAIR['inject'][0]}px;width:{PAIR['inject'][1]-PAIR['inject'][0]}px;color:{PURPLE};border-color:{PURPLE}">injection: attention onto paragraph 2</div>
 <div class="blk" id="bc" style="left:{PAIR['control'][0]}px;width:{PAIR['control'][1]-PAIR['control'][0]}px">{CTRL}</div>
 <div class="blk" id="bq" style="left:1530px;width:745px"><div class="ct">not told about the boost</div>{QCOL}</div>
 <div class="blk" id="bt" style="left:2295px;width:685px"><div class="ct">told about the boost</div>{TCOL}</div>
</div>
{syringe("q", DOSEQ)}{syringe("t", DOSET)}
<img class="vio" src="data:image/png;base64,{b64(VP)}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
// equal-height injection columns: stretch both blocks to the tallest
const hs=['bc','bq','bt'].map(i=>R(i).height); const H=Math.max(...hs);
['bc','bq','bt'].forEach(i=>document.getElementById(i).style.height=H+'px');
document.getElementById('top').style.height=(110+H+40)+'px';
const SW=document.body.scrollWidth, SH=document.body.scrollHeight;
const ov=document.getElementById('ov'), gl=document.getElementById('gl');
[ov,gl].forEach(s=>{{s.setAttribute('width',SW);s.setAttribute('height',SH);}});
let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad,s)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d*s}}" y="${{r.top-d*s}}" width="${{r.width+2*d*s}}" height="${{r.height+2*d*s}}" rx="${{rad+d*s}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
let s=`<defs><marker id="mT" markerUnits="userSpaceOnUse" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="34" markerHeight="34" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PURPLE}"/></marker></defs>`;
[['q','qq','qr',0.72,32,-30],['t','tq','tr',0.72,52,-30]].forEach(([p,qid,rid,sc,ang,xr])=>{{
  const q=R(qid), ro=R(rid); g+=aura(q,32,0.6); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},50,0.6);
  const sy=document.getElementById('syr'+p); const tipx=ro.left+ro.width*0.29, tipy=ro.top+ro.height*0.72;
  sy.style.left=(tipx-380)+'px'; sy.style.top=(tipy-80)+'px'; sy.style.transform=`rotate(${{ang}}deg) scale(${{sc}})`;
  const c=R(p+'c'), t=R(p+'t'), blk=R(p=='q'?'bq':'bt'); const yP=(c.top+c.bottom)/2, xq=blk.right+xr, yq=(q.top+q.bottom)/2;
  s+=`<path d="M${{q.right+2}},${{yq}} L${{xq-22}},${{yq}} Q${{xq}},${{yq}} ${{xq}},${{yq-22}} L${{xq}},${{yP+22}} Q${{xq}},${{yP}} ${{xq-22}},${{yP}} L${{t.right+12}},${{yP}}" fill="none" stroke="{PURPLE}" stroke-width="12" marker-end="url(#mT)"/>`;
}});
gl.innerHTML=g; ov.innerHTML=s;
</script></body></html>'''
NAME = "hero4" if STAT == "mean" else "hero4_median"
hp = os.path.join(HERE, f"_{NAME}.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, f"_{NAME}_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3000",
                "--force-device-scale-factor=1", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 30)); im.save(os.path.join(HERE, NAME + ".png")); print(NAME + ".png", im.size)
os.remove(raw); os.remove(hp); os.remove(VP)
