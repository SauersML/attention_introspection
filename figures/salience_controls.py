"""Salience controls (Qwen3-32B, 240 contexts each, paired per context; reads results/salience_ctrl/data and results/salience_honest/data).
Method strip: the same token-row drawing as salience.py ([P1]..[P4] cells, [P2] amber, question + answer cells, band of layers 21-53) three
times: attention boost x20 (purple), paragraph 2's content added into the residual stream, same size (green), a random direction per layer,
same size (slate).
Top row, the told prompt (only attention boosts mentioned): left, greedy reply starts "Yes" for nothing changed / attention / content / random;
right, top digit after "Yes, paragraph [P" is the boosted paragraph, plus the same attention boost with the question "Pick one of the four
paragraphs at random" (prefill "paragraph [P").
Bottom row, the honest prompt (attention boost, content addition or nothing, a third each): "Yes"; top digit = boosted paragraph;
P(attention) = 1/(1+exp(lp_act-lp_att)) after the prefill "Yes, paragraph [Pk],".
Boosted arms averaged over k = P1..P4 per context. Bars = mean over contexts, whiskers = 95% bootstrap CI (10,000 resamples), p = two-sided
sign-flip over contexts on paired differences (20,000 flips).
usage: OMP_NUM_THREADS=2 nice -n 19 python3 figures/salience_controls.py [--keep]   (writes figures/salience_controls.png and .jpg)"""
import json, glob, os, sys, base64
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
from pfmt import pstr
import injb_draw as dr
ROOT = os.path.dirname(HERE)
GREEN = "#1A9E77"; SLATE = "#556070"; NF = 20000; NB = 10000


def load(d):
    return sorted((json.loads(l) for f in glob.glob(os.path.join(ROOT, f"results/{d}/data/recs_*.jsonl")) for l in open(f)), key=lambda r: r["id"])


yes = lambda t, k: float(t["text"].strip().startswith("Yes"))
top = lambda t, k: float(int(np.argmax(t["lp_par"])) == k)
patt = lambda t, k: 1 / (1 + np.exp(t["lp_act"] - t["lp_att"]))
def arm(recs, a, f):
    return np.array([f(r[a], 0) if "lp_par" in r[a] else np.mean([f(r[a][str(k)], k) for k in range(4)]) for r in recs])


rng = np.random.default_rng(0)
def ci(v):
    b = v[rng.integers(0, len(v), (NB, len(v)))].mean(1); return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
def signflip(d):
    s = rng.choice([-1, 1], (NF, len(d))); null = np.abs((s * d).mean(1)); return float((np.sum(null >= abs(d.mean())) + 1) / (NF + 1))


C, H = load("salience_ctrl"), load("salience_honest")
assert len(C) == 240 and len(H) == 240
X = dict(ctrl=dict(yes={a: arm(C, a, yes) for a in ["none", "A", "B", "B2", "R"]}, top={a: arm(C, a, top) for a in ["A", "B", "B2", "R", "QA"]}),
         honest=dict(yes={a: arm(H, a, yes) for a in ["none", "A", "B"]}, top={a: arm(H, a, top) for a in ["A", "B"]}, patt={a: arm(H, a, patt) for a in ["A", "B"]}))
SC = {e: {rd: {a: ci(v) for a, v in d.items()} for rd, d in E.items()} for e, E in X.items()}
PV = {"ctrl yes A-B": ("ctrl", "yes", "A", "B"), "ctrl yes A-R": ("ctrl", "yes", "A", "R"), "ctrl yes A-none": ("ctrl", "yes", "A", "none"),
      "ctrl yes B-none": ("ctrl", "yes", "B", "none"), "ctrl yes R-none": ("ctrl", "yes", "R", "none"),
      "ctrl top A-B": ("ctrl", "top", "A", "B"), "ctrl top A-QA": ("ctrl", "top", "A", "QA"), "ctrl top B-R": ("ctrl", "top", "B", "R"),
      "honest yes A-B": ("honest", "yes", "A", "B"), "honest yes A-none": ("honest", "yes", "A", "none"), "honest yes B-none": ("honest", "yes", "B", "none"),
      "honest top A-B": ("honest", "top", "A", "B"), "honest patt A-B": ("honest", "patt", "A", "B")}
P = {k: signflip(X[e][rd][a] - X[e][rd][b]) for k, (e, rd, a, b) in PV.items()}
for e in SC:
    for rd in SC[e]: print(e, rd, {a: [round(100 * x, 1) for x in v] for a, v in SC[e][rd].items()})
print({k: f"{v:.2g}" for k, v in P.items()})
json.dump(dict(n=240, rates=SC, p=P), open(os.path.join(os.environ.get("TMPDIR", "/tmp"), "salience_controls_scores.json"), "w"), indent=1)

# ---------- bar panels (one matplotlib image per row) ----------
W = 3000; PAD = 26; VW = W - 2 * PAD
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
NONE = ("none", "white", 1.0, "nothing\nchanged", INK)
LAB = dict(A=(PURPLE, 0.8, "attention\nboost", PURPLE), B=(GREEN, 0.8, "content\nadded", GREEN), R=(SLATE, 0.8, "random\ndirection", SLATE),
           QA=(PURPLE, 0.35, "attention\nboost,\nasked to\npick at random", PURPLE))


def row(path, panels, H=1150, BOT=330, TOPH=200):
    """panels: list of (width share, title, ylim top, [(key, values)], chance or None, chance label, [(i, j, p)], gap-after index)"""
    fig = plt.figure(figsize=(VW / 100, H / 100), dpi=100)
    tot = sum(p[0] for p in panels); GAP = 40; LM = 150; RM = 215
    x0 = 0
    for share, title, ytop, bars, chance, clab, brs, sep in panels:
        pw = (VW - GAP * (len(panels) - 1)) * share / tot
        ax = fig.add_axes([(x0 + LM) / VW, BOT / H, (pw - LM - RM) / VW, (H - BOT - TOPH) / H])
        nb = len(bars) + (0.5 if sep is not None else 0); ax.set_xlim(-0.5, nb - 0.5); ax.set_ylim(0, ytop)
        xs = [i + (0.5 if sep is not None and i > sep else 0) for i in range(len(bars))]
        hi_top = {}
        for i, ((key, v), xc) in enumerate(zip(bars, xs)):
            if key == "none": col, al, lab, tc = NONE[1], 1.0, NONE[3], INK
            else: col, al, lab, tc = LAB[key]
            mu, lo, hi = (100 * x for x in v)
            ax.bar(xc, mu, width=0.72, color=col, alpha=al, lw=4 if key == "none" else 0, edgecolor=INK, zorder=3)
            ax.plot([xc, xc], [lo, hi], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            for yy in (lo, hi): ax.plot([xc - 0.09, xc + 0.09], [yy, yy], color=INK, lw=6, zorder=5, solid_capstyle="butt")
            ax.text(xc, hi + ytop * 0.015, f"{mu:.0f}%", ha="center", va="bottom", fontsize=dr.px(70), fontweight="bold", color=tc)
            hi_top[i] = hi + ytop * 0.14
            fig.text((x0 + LM + (xc + 0.5) / nb * (pw - LM - RM)) / VW, (BOT - 16) / H, lab, ha="center", va="top", fontsize=dr.px(50), fontweight="bold", color=tc, linespacing=1.0)
        if chance is not None:
            ax.plot([-0.5, nb - 0.5], [chance, chance], color=INK, lw=4, ls=(0, (4, 3)), zorder=6)
            ax.text(nb - 0.5 + 0.08, chance, clab, ha="left", va="center", linespacing=1.0, fontsize=dr.px(52), fontweight="bold", color=INK, zorder=7,
                    bbox=dict(fc="white", ec="none", pad=1, alpha=0.85))
        lvl = 0
        for i, j, p in brs:
            yb = max(hi_top[k] for k in range(min(i, j), max(i, j) + 1)) + lvl
            ax.plot([xs[i], xs[i], xs[j], xs[j]], [yb - ytop * 0.025, yb, yb, yb - ytop * 0.025], color=INK, lw=4.5, zorder=5, clip_on=False)
            ax.text((xs[i] + xs[j]) / 2, yb + ytop * 0.01, pstr(p, floor=1 / (NF + 1)), ha="center", va="bottom", fontsize=dr.px(54), color=INK)
            lvl += ytop * 0.14
        for t in range(0, int(ytop) + 1, 20 if ytop <= 80 else 25):
            ax.text(-0.55, t, f"{t}%", ha="right", va="center", fontsize=dr.px(54), fontweight="bold", color=INK)
        ax.plot([-0.5, nb - 0.5], [0, 0], color=INK, lw=4, zorder=4)
        fig.text((x0 + LM + (pw - LM - RM) / 2) / VW, (H - 12) / H, title, ha="center", va="top", fontsize=dr.px(66), fontweight="bold", color=INK, linespacing=1.05)
        ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
        x0 += pw + GAP
    fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


c, h = SC["ctrl"], SC["honest"]
VP1, VP2 = os.path.join(HERE, "_v_salctl1.png"), os.path.join(HERE, "_v_salctl2.png")
row(VP1, [(1.0, "how often Qwen\u2019s reply\nstarts with \u201cYes\u201d", 62, [("none", c["yes"]["none"]), ("A", c["yes"]["A"]), ("B", c["yes"]["B"]), ("R", c["yes"]["R"])],
           None, None, [(1, 2, P["ctrl yes A-B"])], None),
          (1.15, "how often Qwen\u2019s top answer is\nthe boosted paragraph\u2019s number", 80, [("A", c["top"]["A"]), ("B", c["top"]["B"]), ("R", c["top"]["R"]), ("QA", c["top"]["QA"])],
           25, "chance\n25%", [(0, 1, P["ctrl top A-B"]), (0, 3, P["ctrl top A-QA"])], 2)], H=1180, BOT=330, TOPH=260)
row(VP2, [(0.95, "how often Qwen\u2019s reply\nstarts with \u201cYes\u201d", 70, [("none", h["yes"]["none"]), ("A", h["yes"]["A"]), ("B", h["yes"]["B"])],
           None, None, [(1, 2, P["honest yes A-B"])], None),
          (0.75, "top answer is the\nboosted paragraph", 80, [("A", h["top"]["A"]), ("B", h["top"]["B"])], 25, "chance\n25%", [(0, 1, P["honest top A-B"])], None),
          (0.75, "how likely Qwen\nsays \u201cattention\u201d", 100, [("A", h["patt"]["A"]), ("B", h["patt"]["B"])], 50, "chance\n50%", [(0, 1, P["honest patt A-B"])], None)], H=1100, BOT=230)


# ---------- method drawings ----------
def stream(kind):
    """token row ([P1]..[P4], question + answer) under the band of layers 21-53; kind 'att', 'add' or 'rand'"""
    w, h = 950, 560; o = []; cw, cg = 40, 6
    groups = [("P1", 3), ("P2", 3), ("P3", 3), ("P4", 3), ("Q", 5)]
    x = 28; cells = {}
    for g, n in groups:
        if g == "Q": x += 22
        cells[g] = [x + j * (cw + cg) for j in range(n)]; x += n * (cw + cg) + 14
    by0, by1, ty = 150, 380, 400
    o.append(f'<rect x="12" y="{by0}" width="{w - 24}" height="{by1 - by0}" rx="24" fill="#F4F4F6" stroke="{INK}" stroke-width="4"/>')
    o.append(f'<text x="34" y="{by0 + 60}" font-size="48" font-weight="800" fill="{INK}">layers 21\u201353</text>')
    for g, xs in cells.items():
        for cx in xs:
            fill = dr.AMB if g == "P2" else ("#1F8BFF" if g == "Q" else "#9DC9FF")
            o.append(f'<rect x="{cx}" y="{ty}" width="{cw}" height="56" rx="8" fill="{fill}"/>')
            o.append(f'<line x1="{cx + cw / 2}" y1="{ty}" x2="{cx + cw / 2}" y2="{by0 + 90 if g != "Q" else by0 + 4}" stroke="{INK}" stroke-opacity=".18" stroke-width="3"/>')
        lab = "question" if g == "Q" else f"[{g}]"
        o.append(f'<text x="{(xs[0] + xs[-1] + cw) / 2}" y="{ty + 110}" font-size="46" font-weight="800" text-anchor="middle" fill="{INK}">{lab}</text>')
    p2 = (cells["P2"][0] + cells["P2"][-1] + cw) / 2; qs = [c_ + cw / 2 for c_ in cells["Q"]]; yp = by0 + 170
    if kind == "att":
        o.append(f'<defs><marker id="ha" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="{PURPLE}"/></marker></defs>')
        for j, qx in enumerate(qs[::2]):
            y0 = by1 - 36 - 20 * j
            o.append(f'<path d="M{qx},{y0} C{qx},{by0 + 70} {p2 + 50},{by0 + 70} {p2 + 8 * j - 10},{by1 - 28}" fill="none" stroke="{PURPLE}" stroke-width="11" stroke-opacity=".8" marker-end="url(#ha)"/>')
        o.append(f'<text x="{w / 2}" y="{by0 - 30}" font-size="58" font-weight="800" text-anchor="middle" fill="{PURPLE}">attention to [P2] \u00d720</text>')
    else:
        col = GREEN if kind == "add" else SLATE; mid = "hg" if kind == "add" else "hs"
        if kind == "add":
            o.append(f'<defs><marker id="{mid}" markerWidth="5" markerHeight="5" refX="2.5" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5 Z" fill="{col}"/></marker></defs>')
            o.append(f'<path d="M{p2},{ty - 6} C{p2},{by0 + 80} {qs[0] - 100},{by0 + 80} {qs[0] - 40},{yp - 8}" fill="none" stroke="{col}" stroke-width="13" marker-end="url(#{mid})"/>')
        for qx in qs:
            o.append(f'<circle cx="{qx}" cy="{yp}" r="19" fill="#fff" stroke="{col}" stroke-width="6"/><path d="M{qx - 10},{yp} H{qx + 10} M{qx},{yp - 10} V{yp + 10}" stroke="{col}" stroke-width="6"/>')
        txt = "+ [P2]\u2019s content, same size" if kind == "add" else "+ random direction, same size"
        o.append(f'<text x="{w / 2}" y="{by0 - 30}" font-size="58" font-weight="800" text-anchor="middle" fill="{col}">{txt}</text>')
    return f'<svg width="{w}" height="{h - 60}" viewBox="0 60 {w} {h - 60}">{"".join(o)}</svg>'


b64 = lambda p: base64.b64encode(open(p, "rb").read()).decode()
TOLD = "I am capable of artificially boosting your attention onto one of the four paragraphs below"
assert TOLD in open(os.path.join(ROOT, "results/salience/data/meta.json")).read()
HP = open(os.path.join(ROOT, "results/salience_honest/data/example_prompt_0.txt")).read()
for s_ in ["either I boost your attention onto that paragraph's text, or I add that paragraph's content directly into your activations",
           "A third of trials are attention boosts, a third are activation additions, and a third are control trials",
           "Did I change how you process one of the paragraphs? If so, which one, and did I boost your attention or add to your activations?"]:
    assert s_ in HP, s_
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:16px {PAD}px 10px}}
.ms{{display:flex;justify-content:space-between}}
.hd{{font-size:74px;font-weight:800;margin-top:26px;line-height:1.05}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:18px 30px;font-size:52px;line-height:1.16;margin-top:14px}}
img{{display:block;width:{VW}px;margin-top:6px}}
</style></head><body>
<div class="ms"><div>{stream("att")}</div><div>{stream("add")}</div><div>{stream("rand")}</div></div>
<div class="hd">Qwen3-32B told only about attention boosts</div>
<div class="ub">{TOLD} \u2026 Do you notice your attention being boosted onto one of the paragraphs? If so, which one?</div>
<img src="data:image/png;base64,{b64(VP1)}">
<div class="hd">Qwen3-32B told both kinds are possible</div>
<div class="ub">\u2026 either I boost your attention onto that paragraph\u2019s text, or I add that paragraph\u2019s content directly into your activations \u2026 a third of trials are attention boosts, a third are activation additions, and a third are control trials \u2026 Did I change how you process one of the paragraphs? If so, which one, and did I boost your attention or add to your activations?</div>
<img src="data:image/png;base64,{b64(VP2)}">
</body></html>'''
dr.render(HTML, HERE, "salience_controls", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP1); os.remove(VP2)
