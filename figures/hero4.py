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
Style as hero_L28 / hero_inject: violins outlined purple (rise) or red (drop), dots coloured by their own % change (pal.pct_colour),
dashed baselines, injection columns with the injected robot, aura, syringe (x measured dose) and the wave arrow to paragraph 2.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 hero4.py [mean|median]"""
import json, glob, os, sys, html, base64, subprocess
import numpy as np, matplotlib
import matplotlib.patheffects as pe
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, binomtest, wilcoxon
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from pal import INK, PURPLE, pct_colour
from pfmt import pstr
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
    if STAT == "mean": return pstr(s["p_flip"], floor=1 / (s["nf"] + 1))
    return pstr(s["p_sign_split"])
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

# ---------------- layout (CSS px; the page is laid out 3000 px wide and rendered at 2300 px) ----------------
W = 3000; OUTW = 2300
CC = {"focus": 540, "ignore": 1110, "question": 1840, "told": 2600}          # violin centres
AXX = {"control": 236, "inject": 1592}                                         # y axes
PAIR = {"control": (20, 1330), "inject": (1380, 2975)}
UP = {k: cen(Z[k]) > 0 for k in Z}
COLR = {k: (PURPLE if UP[k] else RED) if round(cen(Z[k])) else INK for k in Z}   # a 0% centre in black                               # rises purple, drops red (as hero_L28)

# ---------------- result row (matplotlib, pixel coordinates) ----------------
VH = 1250; PT = lambda v: v * 72 / 100   # css px -> points at dpi 100
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(W / 100, VH / 100), dpi=100)
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(VH, 0); ov.axis("off")
PY0, PY1 = 500, VH - 30                       # plot band (px from top)
def ymap(lo, hi): return lambda v: PY1 - (PY1 - PY0) * (v - lo) / (hi - lo)
LC = np.log2(1 + np.concatenate([Z["focus"], Z["ignore"]]) / 100); YC = ymap(LC.min() - 0.2, LC.max() + 0.2)
YI = ymap(-100 - 56, 300 + 56)
DOT = 840   # dot area, points^2 (hero: big translucent dots with a black speck)
def violin(k, Y, vals, ratio, jit, bw, halfw):
    z = vals; kde = gaussian_kde(z, bw_method=bw / z.std(ddof=1))
    yy = np.linspace(z.min() - 1.5 * bw, z.max() + 1.5 * bw, 500); dd = kde(yy); dd = dd / dd.max() * halfw; col = PURPLE if cen(z) >= 0 else RED   # violin: purple unless the centre drops
    ov.fill_betweenx([Y(v) for v in yy], CC[k] - dd, CC[k] + dd, color=col, alpha=0.18, lw=0)
    ov.plot(np.r_[CC[k] - dd, (CC[k] + dd)[::-1], CC[k] - dd[:1]], [Y(v) for v in np.r_[yy, yy[::-1], yy[:1]]], color=col, lw=3.2)
    rng = np.random.default_rng(1); zj = z + rng.uniform(-jit, jit, len(z))
    xj = CC[k] + rng.uniform(-1, 1, len(z)) * kde(zj) / kde(yy).max() * halfw * 0.9
    ov.scatter(xj, [Y(v) for v in zj], s=DOT, c=pct_colour(ratio), alpha=0.45, edgecolors="none", zorder=3)
    ov.scatter(xj, [Y(v) for v in zj], s=7, color="black", zorder=4)
for k in ("focus", "ignore"):
    violin(k, YC, np.log2(1 + Z[k] / 100), 1 + Z[k] / 100, 0, 0.12, 230)
for k in ("question", "told"):
    violin(k, YI, Z[k], 1 + Z[k] / 100, 16, 36, 205)
# dashed baselines, axes, ticks (one y axis per pair: the two pairs measure different things)
for pr, Y, ticks, lab in [("control", YC, [(-1, "\u221250%"), (0, "0%"), (1, "+100%"), (2, "+300%"), (3, "+700%")], "only mentioning\nparagraph 2"),
                          ("inject", YI, [(t, neg(f"{t:+d}%") if t else "0%") for t in (-100, 0, 100, 200, 300)], "no injection")]:
    x0, x1 = PAIR[pr]; ax_ = AXX[pr]
    ov.plot([ax_ + 10, x1], [Y(0)] * 2, color=INK, lw=5, ls=(0, (4, 3)), zorder=5)
    for v, t in ticks:
        ov.text(ax_, Y(v), t, ha="right", va="center", fontsize=PT(66), fontweight="bold", color=INK)
    if pr == "control": ov.text(ax_ + 30, Y(0) + 14, lab, ha="left", va="top", fontsize=PT(66), color=INK, zorder=6, linespacing=1.05)
    else: ov.text((CC["question"] + CC["told"]) / 2 + 8, Y(0) - 12, lab.replace(" ", "\n"), ha="center", va="bottom", fontsize=PT(66), color=INK, linespacing=1.05, zorder=6)
# black centre lines, their label once per pair
for k, Y, tr in [("focus", YC, lambda v: np.log2(1 + v / 100)), ("ignore", YC, lambda v: np.log2(1 + v / 100)),
                 ("question", YI, lambda v: v), ("told", YI, lambda v: v)]:
    y = Y(tr(cen(Z[k]))); ov.plot([CC[k] - 130, CC[k] + 130], [y, y], color=INK, lw=6, zorder=6, solid_capstyle="butt")
# axis titles (one per pair) and big numbers
YT = {"control": "change in attention from the answer\nto paragraph 2",
      "inject": "change in how often Qwen\u2019s\nanswer is the boosted paragraph"}
for pr in YT:
    ov.text((PAIR[pr][0] + PAIR[pr][1]) / 2, 8, YT[pr], ha="center", va="top", fontsize=PT(66), fontweight="bold", color=INK, linespacing=1.1)
for k in CC:
    v = cen(Z[k]); s = neg(f"{v:+.0f}%") if round(v) else "0%"
    ov.text(CC[k], 345, s, ha="center", va="baseline", fontsize=PT(130), fontweight="bold", color=COLR[k])
    ov.text(CC[k], 368, ptxt(k), ha="center", va="top", fontsize=PT(74), color=COLR[k])
# key, inside the empty region below zero in the focus column
ky = YC(2.75); kx = CC["focus"] + 250
ov.scatter([kx], [ky], s=DOT, color=pct_colour(2.0), alpha=0.6, edgecolors="none"); ov.scatter([kx], [ky], s=7, color="black")
ov.text(kx + 36, ky, "= one prompt", ha="left", va="center", fontsize=PT(66), fontweight="bold", color=INK)
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
def arcs_svg(w=1310, h=460):
    """what the control measures: attention from the answer tokens to paragraph 2's tokens (a drawing of the measure, not values)"""
    o = []; sq, gp, gg, x, ty = 20, 5, 30, 30, 340; xs = []
    for g in range(4):
        gx = []
        for t in range(8):
            o.append(f'<rect x="{x}" y="{ty}" width="{sq}" height="{sq + 10}" rx="4" fill="{AMB if g == 1 else "#C9CCD3"}"/>'); gx.append(x + sq / 2); x += sq + gp
        if g == 1: o.append(f'<text x="{(gx[0] + gx[-1]) / 2:.0f}" y="{ty + sq + 80}" font-size="60" font-weight="700" fill="{INK}" text-anchor="middle">paragraph 2</text>')
        xs.append(gx); x += gg
    rx_ = x + 30; o.append(f'<image href="data:image/png;base64,{robot}" x="{rx_}" y="{ty + 18 - 65}" width="130" height="130"/>')
    bx = rx_ + 146; bw, bh = 170, 96; by_ = ty + 18 - bh / 2
    o.append(f'<rect x="{bx}" y="{by_}" width="{bw}" height="{bh}" rx="30" fill="#fff" stroke="#D9D9DE" stroke-width="3"/>')
    o.append(f'<text x="{bx + bw / 2}" y="{by_ + bh / 2 + 24}" font-size="68" font-weight="700" fill="{INK}" text-anchor="middle">{html.escape(ANS_C)}</text>')
    sx, sy = bx + bw / 2, by_ - 4
    for tx in xs[1]:
        o.append(f'<path d="M{sx:.0f},{sy:.0f} C{sx - 60:.0f},{ty - 290} {tx + 40:.0f},{ty - 290} {tx:.0f},{ty - 6}" fill="none" stroke="{INK}" stroke-width="4" stroke-linecap="round"/>')
    o.append(f'<text x="{(xs[1][-1] + sx) / 2 + 40:.0f}" y="{ty - 268}" font-size="60" font-weight="700" fill="{INK}" text-anchor="middle">attention at layer 28</text>')
    vw = x + 30 + 146 + 170 + 10   # drawn at its natural width, shown across the column
    return f'<svg width="{w}" height="{h * w / vw:.0f}" viewBox="0 0 {vw} {h}" font-family="Helvetica Neue,Helvetica,Arial,sans-serif">{"".join(o)}</svg>'

CTRL = f'''<div class="ub" style="font-size:68px">Here are four paragraphs.{plc}
  <div class="vrow"><div class="var" style="color:{COLR["focus"]}">{html.escape(LINE["3_focus"])}</div><div class="var" style="color:{COLR["ignore"]}">{html.escape(LINE["9_completely_ignore"])}</div></div>
  <div>{html.escape(qline_c)}</div></div>
  <div style="margin-top:auto">{arcs_svg()}</div>'''
# told intro: the run's real intro text, shortened with ellipses (as hero_inject.py)
TMETA = json.load(open(os.path.join(ROOT, "results/told_inject/data/meta.json")))
S2 = "I am capable of artificially boosting your attention onto one of the four paragraphs below"; S3 = "on 50% of trials"
for s_ in (S2, S3): assert s_ in TMETA["intro"], s_
assert TMETA["question"].startswith("Trial 1: Do you notice your attention being boosted onto one of the paragraphs? If so, which one?")
QCOL = f'''<div class="ub i" id="qb">Here are four paragraphs.{plines(EXQ["segs"], EXQ["lab"], KQ, "q", 2)}<div>{html.escape(QLINE)}</div></div>
  <div class="mrow" style="margin-left:30px"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(rq["ans"])}</div></div>
  <div class="ub2" id="qq">Which paragraph did you attend to most while writing your answer? \u2026</div>
  <div class="mrow inj" style="margin-left:250px"><img class="rob" id="qr" src="data:image/png;base64,{robot_inj}"><div class="mb" id="qm">{html.escape(QREPLY)}</div></div>'''
TCOL = f'''<div class="ub i" id="tb">\u2026 {S2} \u2026 {S3}.{plines(segs_t, [1, 2, 3, 4], KT, "t", 2)}</div>
  <div class="ub2" id="tq">Do you notice your attention being boosted onto one of the paragraphs? If so, which one?</div>
  <div class="mrow inj" style="margin-left:0"><img class="rob" id="tr" src="data:image/png;base64,{robot_inj}"><div class="mb" id="tm">{html.escape(TREPLY)}</div></div>'''
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};position:relative;padding-top:18px}}
.hd{{position:absolute;top:14px;font-size:78px;font-weight:800;white-space:nowrap}}
.top{{position:relative;height:var(--th)}}
.blk{{position:absolute;top:118px;display:flex;flex-direction:column}}
.ct{{font-size:62px;font-weight:800;color:{INK};margin:0 0 12px 6px;white-space:nowrap}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:20px 28px;font-size:60px;line-height:1.18}}
.pl{{white-space:nowrap;overflow:hidden}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.vrow{{display:flex;gap:22px;margin:10px 0}}
.var{{flex:1;background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.15}}
.blk .ub.i{{margin-right:96px}}
.ub2{{background:#1F8BFF;color:#fff;border-radius:34px;padding:16px 26px;font-size:60px;line-height:1.18;margin:64px 0 0 70px;position:relative;z-index:2}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:16px}}
.mrow.inj{{margin-top:auto;padding-top:120px}}
.rob{{width:130px;height:130px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:8px 26px;font-size:66px;font-weight:700;white-space:nowrap;box-shadow:0 3px 8px rgba(0,0,0,.08);position:relative;z-index:2}}
.syr{{position:absolute;transform-origin:380px 80px;z-index:3}}
#gl{{position:absolute;left:0;top:0;z-index:-1;pointer-events:none}} #ov{{position:absolute;left:0;top:0;pointer-events:none;z-index:1}}
.vio{{width:{W}px;display:block;margin-top:40px}}
</style></head><body>
<div class="top" id="top">
 <div class="hd" style="left:{PAIR['control'][0] + 4}px">instructions in the prompt</div>
 <div class="hd" style="left:{PAIR['inject'][0] + 4}px;color:{PURPLE}">inject attention toward paragraph 2</div>
 <div class="blk" id="bc" style="left:{PAIR['control'][0]}px;width:{PAIR['control'][1]-PAIR['control'][0]}px">{CTRL}</div>
 <div class="blk" id="bq" style="left:1380px;width:750px"><div class="ct">not told about it</div>{QCOL}</div>
 <div class="blk" id="bt" style="left:2170px;width:780px"><div class="ct">told about it</div>{TCOL}</div>
</div>
{syringe("q", DOSEQ)}{syringe("t", DOSET)}
<img class="vio" src="data:image/png;base64,{b64(VP)}">
<svg id="gl"></svg><svg id="ov"></svg>
<script>
const R=e=>document.getElementById(e).getBoundingClientRect();
// equal-height columns: stretch the blocks to the tallest
const hs=['bc','bq','bt'].map(i=>R(i).height); const H=Math.max(...hs);
['bc','bq','bt'].forEach(i=>document.getElementById(i).style.height=H+'px');
document.getElementById('top').style.height=(118+H)+'px';
const SW=document.body.scrollWidth, SH=document.body.scrollHeight;
const ov=document.getElementById('ov'), gl=document.getElementById('gl');
[ov,gl].forEach(s=>{{s.setAttribute('width',SW);s.setAttribute('height',SH);}});
let g=`<defs>`;
[[0.004,0.010,36,7,1],[0.006,0.014,28,4.5,2],[0.008,0.02,18,2.5,3]].forEach(([fx,fy,sc,bl,i])=>{{
 g+=`<filter id="w${{i}}" x="-30%" y="-30%" width="160%" height="160%"><feTurbulence type="turbulence" baseFrequency="${{fx}} ${{fy}}" numOctaves="2" seed="${{5*i+1}}" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="${{sc}}" xChannelSelector="R" yChannelSelector="G" result="d"/><feGaussianBlur in="d" stdDeviation="${{bl}}"/></filter>`;}});
g+=`<filter id="haze" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="22"/></filter></defs>`;
const aura=(r,rad,s)=>{{ const S=(d,st,col,op,f,fill)=>`<rect x="${{r.left-d*s}}" y="${{r.top-d*s}}" width="${{r.width+2*d*s}}" height="${{r.height+2*d*s}}" rx="${{rad+d*s}}" fill="${{fill?col:'none'}}" stroke="${{fill?'none':col}}" stroke-width="${{st}}" opacity="${{op}}" filter="url(#${{f}})"/>`;
 return S(8,0,'#B994FF',0.75,'haze',1)+S(50,5,'#C9A8FF',0.45,'w1')+S(32,7,'{LIL}',0.6,'w2')+S(15,9,'#7B3FE4',0.75,'w3')+S(2,8,'{PURPLE}',0.55,'w3'); }};
// light-purple wave arrow (hero_inject.py): one cubic, filled with wave bands that flow toward the tip
const SW0=12;
let NA=0;
const arrow=(p0,c1,c2,tip)=>{{ const SW=SW0*1.7; const L=SW*2.7, H=SW*1.6; let ux=tip[0]-c2[0], uy=tip[1]-c2[1]; const n=Math.hypot(ux,uy); ux/=n; uy/=n;
 const e=[tip[0]-L*ux, tip[1]-L*uy], b=[e[0]+1.5*ux, e[1]+1.5*uy];
 const d=`M${{p0}} C${{c1}} ${{c2}} ${{e}}`, h=`M${{tip}} L${{b[0]-H*uy}},${{b[1]+H*ux}} L${{b[0]+H*uy}},${{b[1]-H*ux}} Z`;
 const o=`<path d="${{d}}" fill="none" stroke="{INK}" stroke-width="${{SW+12}}" stroke-linecap="round"/><path d="${{h}}" fill="{INK}" stroke="{INK}" stroke-width="12" stroke-linejoin="round"/>`;
 const B=(t,i)=>{{const u=1-t; return u*u*u*p0[i]+3*u*u*t*c1[i]+3*u*t*t*c2[i]+t*t*t*e[i];}};
 const P=[]; const N=600;
 for (let k=0;k<=N;k++){{ const t=k/N; P.push([B(t,0),B(t,1)]); }}
 for (let k=1;k<=40;k++) P.push([e[0]+ux*L*1.15*k/40, e[1]+uy*L*1.15*k/40]);
 const S=[0]; for (let k=1;k<P.length;k++) S.push(S[k-1]+Math.hypot(P[k][0]-P[k-1][0],P[k][1]-P[k-1][1]));
 const lam=SW*1.9, half=H*1.05, rows=8, bend=SW*0.9;
 const lerp=(a,b,f)=>Math.round(a+(b-a)*f), C1=[0xE6,0xDB,0xF7], C2=[0x9C,0x7B,0xD8];
 let q='';
 for (let k=0;k<P.length-1;k++){{
  const a=P[Math.max(k-1,0)], bb=P[Math.min(k+2,P.length-1)]; let tx=bb[0]-a[0], ty=bb[1]-a[1]; const m=Math.hypot(tx,ty)||1; tx/=m; ty/=m; const nx=-ty, ny=tx;
  const ds=Math.hypot(P[k+1][0]-P[k][0],P[k+1][1]-P[k][1])+0.8;
  for (let r=0;r<rows;r++){{
   const w0=-half+2*half*r/rows-0.7, w1=-half+2*half*(r+1)/rows+0.7, wm=(w0+w1)/2;
   const ph=(S[k] + bend*Math.pow(Math.abs(wm)/half,1.6))/lam;
   const f=0.5+0.5*Math.sin(2*Math.PI*ph);
   const col=`rgb(${{lerp(C1[0],C2[0],f)}},${{lerp(C1[1],C2[1],f)}},${{lerp(C1[2],C2[2],f)}})`;
   const x=P[k][0], y=P[k][1];
   q+=`<path d="M${{(x+nx*w0).toFixed(1)}},${{(y+ny*w0).toFixed(1)}} L${{(x+nx*w1).toFixed(1)}},${{(y+ny*w1).toFixed(1)}} L${{(x+nx*w1+tx*ds).toFixed(1)}},${{(y+ny*w1+ty*ds).toFixed(1)}} L${{(x+nx*w0+tx*ds).toFixed(1)}},${{(y+ny*w0+ty*ds).toFixed(1)}} Z" fill="${{col}}" stroke="${{col}}" stroke-width="0.6"/>`; }} }}
 const xs=P.map(p=>p[0]), ys=P.map(p=>p[1]), pad=SW*4;
 const X=Math.min(...xs)-pad, Y=Math.min(...ys)-pad, Wd=Math.max(...xs)-X+pad, Ht=Math.max(...ys)-Y+pad; NA+=1;
 return o+`<defs><mask id="am${{NA}}" maskUnits="userSpaceOnUse" x="${{X}}" y="${{Y}}" width="${{Wd}}" height="${{Ht}}"><path d="${{d}}" fill="none" stroke="#fff" stroke-width="${{SW}}" stroke-linecap="round"/><path d="${{h}}" fill="#fff"/></mask></defs><g mask="url(#am${{NA}})">${{q}}</g>`; }};
let s='';
[['q','qq','qr','qm',0.72,34],['t','tq','tr','tm',0.72,34]].forEach(([p,qid,rid,mid,sc,ang])=>{{
  const q=R(qid), ro=R(rid); g+=aura(q,34,0.6); g+=aura(R(mid),30,0.6); g+=aura({{left:ro.left+6,top:ro.top+10,width:ro.width-12,height:ro.height-16}},50,0.6);
  const sy=document.getElementById('syr'+p); const tipx=ro.left+ro.width*0.29, tipy=ro.top+ro.height*0.72;
  sy.style.left=(tipx-380)+'px'; sy.style.top=(tipy-80)+'px'; sy.style.transform=`rotate(${{ang}}deg) scale(${{sc}})`;
  // where the injected attention goes: up out of the question's top edge, a quarter-turn, into the right end of [P2]
  // up the gutter right of the paragraph bubble, then a tight turn left along [P2]'s line into its right end
  const t=R(p+'t'), pb=R(p=='q'?'qb':'tb'); const gx=(pb.right+q.right)/2, P0=[gx, q.top+30], T=[t.right+14, (t.top+t.bottom)/2];
  s+=arrow(P0, [gx, T[1]+40], [gx-40, T[1]], T);
}});
gl.innerHTML=g; ov.innerHTML=s;
</script></body></html>'''
NAME = "hero4" if STAT == "mean" else "hero4_median"
hp = os.path.join(HERE, f"_{NAME}.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, f"_{NAME}_raw.png")
SC = OUTW / W   # lay out at 3000 CSS px, screenshot at 2300 px
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3400",
                f"--force-device-scale-factor={SC:.6f}", "--virtual-time-budget=3000", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); print("raw", im.size)
if im.size[0] != OUTW: im = im.resize((OUTW, round(im.size[1] * OUTW / im.size[0])), Image.LANCZOS)
ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, OUTW, nz[-1] + 24)); im.save(os.path.join(HERE, NAME + ".png")); print(NAME + ".png", im.size, "aspect", round(im.size[1] / im.size[0], 2))
os.remove(raw); os.remove(hp); os.remove(VP)
