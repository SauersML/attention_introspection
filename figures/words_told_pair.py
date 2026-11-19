"""words_told_pair: told_pair.py's design for the told-injection run on single-word lists (Qwen3-32B, 240 contexts; results/told_inject_words,
told_inject_words/told_inject_words.py). Each context: the researcher's intro, eight single-token nouns from eight categories, one per line,
labelled [W1]..[W8], and an arithmetic line; Qwen's real unboosted answer; then the second user turn "Trial 1: Do you notice your attention
being boosted onto one of the words? If so, which one?". The boost adds log(multiplier) to the pre-softmax attention scores toward the boosted
word's token from every query from "Trial 1" on, layers 21-53. Top: the method drawing from injb_draw.py (example context nearest the median of
P(boosted digit) at multiplier 10 among those whose top prefilled digit is the boosted word, word 3 boosted; Qwen's real greedy reply "No"
and the prefilled "Yes, word [W" with the real top digit, black arrows to the two panel headers).
LEFT: greedy replies starting with "Yes": 0 of 240 control and 0 of 7,680 boosted trials (240 contexts x 8 words x 4 doses), so the relative
change from no boost is undefined; the panel shows the raw rates (0% at every dose; no-boost rate 0%) and says so.
RIGHT: top answer after the prefill "Yes, word [W" (boost on) = argmax of the stored log-probs of digits 1-8; per context, the percent of the
8 boosted words where the top answer is the boosted word's number, relative to no boost (exactly 12.5% for every context: one fixed top answer
matches 1 of 8 rotations); whiskers 95% bootstrap over contexts (20,000 resamples); p: two-sided sign-flip over contexts at the top dose
(200,000 flips). Dose labels: measured attention mass on the boosted word's token, boost / no boost, mean over contexts and words.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 words_told_pair.py [--keep]   (writes figures/words_told_pair.png and .jpg)"""
import json, glob, os, sys, random
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import INK, PURPLE
import injb_draw as dr
import told_common as tc
ROOT = os.path.dirname(HERE); D = os.path.join(ROOT, "results/told_inject_words/data/")
R = sorted([json.loads(l) for f in glob.glob(D + "recs_*.jsonl") for l in open(f) if l.strip()], key=lambda r: r["id"])
META = json.load(open(D + "meta.json"))
NW = 8; M = 10.0; K = 2                     # example: word 3 boosted at multiplier 10
W = 3600; PAD = 26; NF = 200000
MS = [2.0, 5.0, 10.0, 20.0]
assert len(R) == 240
T = lambda r, k, m: r["boost"][f"{k}|{m}"]
yes = lambda t: t.strip().lower().startswith("yes")

MASSES = {m: float(np.mean([np.mean([T(r, k, m)["mass"][k] / r["none"]["mass"][k] for k in range(NW)]) for r in R])) for m in MS}
# left: greedy Yes
YC = np.array([yes(r["none"]["text"]) for r in R], float)
YB = {m: np.array([np.mean([yes(T(r, k, m)["text"]) for k in range(NW)]) for r in R]) for m in MS}
NY = sum(int(round(NW * YB[m].sum())) for m in MS); NT = len(R) * NW * len(MS)
assert NY == 0 and YC.sum() == 0, (NY, YC.sum())   # Qwen never says Yes: relative change undefined, raw rates drawn
# right: top answer after the prefill = argmax over the eight digit log-probs
AC = np.array([np.mean([np.argmax(r["none"]["lp_par"]) == k for k in range(NW)]) for r in R])
assert np.all(AC == 1 / NW)
AB = {m: np.array([np.mean([np.argmax(T(r, k, m)["lp_par"]) == k for k in range(NW)]) for r in R]) for m in MS}
PN = tc.signflip(AB[20.0] - 1 / NW, NF)
RN = {m: (AB[m].mean() * NW - 1, tc.boot(lambda ix, m=m: AB[m][ix].mean() * NW - 1, len(R))) for m in MS}
print("Yes:", f"{NY} of {NT:,} boosted, {int(YC.sum())} of {len(R)} control")
print("attention x:", [f"{MASSES[m]:.2f}" for m in MS])
print("top answer %:", [f"{100*AB[m].mean():.1f}" for m in MS], "relative:", [f"{100*RN[m][0]:+.0f} [{100*RN[m][1][0]:+.0f}, {100*RN[m][1][1]:+.0f}]" for m in MS], "p", PN)
# example context
PB = np.array([np.exp(T(r, K, M)["lp_par"][K]) / np.exp(T(r, K, M)["lp_par"]).sum() for r in R]); med = np.median(PB)
cand = [i for i, r in enumerate(R) if np.argmax(T(r, K, M)["lp_par"]) == K]
rec = R[min(cand, key=lambda i: abs(PB[i] - med))]; EXID = rec["id"]
EXTXT = T(rec, K, M)["text"].strip(); PICK = int(np.argmax(T(rec, K, M)["lp_par"])) + 1
assert EXTXT == "No" and PICK == K + 1
# the example's arithmetic line: privileged_words.py's own draws (seed 11), replayed from its saved word pool (as in hero_inject_words.py)
PM = json.load(open(os.path.join(ROOT, "results/privileged_words/data/meta.json"))); rr = random.Random(META["seed"])
for i in range(EXID + 1):
    cats = rr.sample(sorted(PM["cats"]), 8); xx = rr.randrange(10, 90); yy = rr.randrange(10, 90)
    ws = [rr.choice(PM["cats"][c]) for c in cats]; rr.sample(range(8), 8); rr.sample(range(1, 9), 8)
assert ws == rec["words"] and str(xx + yy) in rec["ans"], (ws, rec["words"], rec["ans"])
QLINE = META["qline"].format(x=xx, y=yy)
print("example context", EXID, rec["words"], QLINE, "->", rec["ans"].strip(), "| reply", EXTXT, "| prefill top digit", PICK, f"P={PB[R.index(rec)]:.2f} median {med:.2f}")
sys.exit() if "--nums" in sys.argv else None

# ---------------- two panels in one image ----------------
VW = W - 2 * PAD; GAP = 60; LM = 400; VH = 1480; TOPH = 330; BOT = 280
PWL = 1600; PWR = VW - GAP - PWL   # the right panel wider: its big labels ("+388%") need wider columns than the left panel's "0%"
CWL, CWR = (PWL - LM) / 4, (PWR - LM) / 4
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
def axis(lo, hi):   # full-width axes (x in canvas px), one per panel so each has its own y scale
    a = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH]); a.set_xlim(0, VW); a.set_ylim(lo, hi); a.patch.set_visible(False)
    a.set_xticks([]); a.set_yticks([]); [s_.set_visible(False) for s_ in a.spines.values()]; return a
L = dict(PW=PWL, LM=LM, CW=CWL, VW=VW, VH=VH, ylo=-0.10, yhi=1.0); Rg = dict(PW=PWR, LM=LM, CW=CWR, VW=VW, VH=VH, ylo=-0.45, yhi=4.5)
axL = axis(L["ylo"], L["yhi"]); axR = axis(Rg["ylo"], Rg["yhi"])
# left: raw rates (relative change undefined: no Yes anywhere, control included)
tc.bars_panel(fig, axL, 0, L, MS, {m: (YB[m].mean(), None) for m in MS}, YC.mean(), {m: YB[m].mean() for m in MS}, MASSES,
              (0.5, 1.0), "word", ylab="% of replies", zero="0% =\nno boost", tick_lab=lambda t: f"{100*t:.0f}%", big=lambda v: f"{100*v:.0f}%")
axL.text(LM + 2 * CWL, 0.42, f"never says \u201cYes\u201d\n(0 of {NT:,})", ha="center", va="center", fontsize=dr.px(100), fontweight="bold", color=PURPLE, linespacing=1.1)
tc.bars_panel(fig, axR, PWL + GAP, Rg, MS, RN, 1 / NW, {m: AB[m].mean() for m in MS}, MASSES, (1, 2, 3, 4), "word", PN, NF,
              base_lab=lambda b: f"{100*b:.1f}%")
VP = os.path.join(HERE, "_v_words_told_pair.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_words_bubble(META, rec["words"], K, QLINE)
pre = f'<div class="arow" id="rowA"><img class="rob0" src="data:image/png;base64,ROBOT_PLAIN"><div class="mb0">{rec["ans"].strip()}</div></div>'
reply = f'<span class="part" id="ry">No</span>'
titles = (f'<div class="cols"><div class="ct" style="flex:none;width:{PWL - LM:.0f}px;margin-left:{LM}px"><span class="cn" id="c1">how often Qwen\u2019s reply<br>starts with \u201cYes\u201d</span></div>'
          f'<div class="ct" style="flex:none;width:{PWR - LM:.0f}px;margin-left:{GAP + LM:.0f}px"><span class="cn" id="c2">how often Qwen\u2019s top answer is<br>the boosted word\u2019s number</span></div></div>')
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the words? If so, which one?", reply, titles, VP, VW, f"{MASSES[M]:.1f}",
               pre=pre, inj=f"boost attention<br>toward word {K + 1}", arrows=(("ry", "c1"), ("rp", "c2")), tipdrop=-2, inj_px=105, inj_near=(-18, -60))
mb2 = f'<div class="mb" id="mb2"><span class="pfb">Yes, word [W</span><span class="part" id="rp">{PICK}</span>]</div>'
HTML = HTML.replace('<span class="part" id="ry">No</span></div>', '<span class="part" id="ry">No</span></div>' + mb2, 1)
HTML = HTML.replace("g+=aura(R('mb'),30);", "g+=aura(R('mb'),30); g+=aura(R('mb2'),30);", 1)
HTML = HTML.replace("</style>", ".ub{margin-right:700px;font-size:62px}#rq{margin-left:1500px;margin-right:500px;font-size:62px}.mrow{margin-left:1250px;gap:40px}.mb{font-size:96px}.cn{font-size:86px}.cols{margin-top:250px}"
                    f".pfb{{border:5px dashed {INK};border-radius:14px;padding:0 10px;margin-right:6px;position:relative}}"
                    ".pfb::after{content:'prefilled';position:absolute;left:50%;transform:translateX(-50%);top:100%;margin-top:30px;font-size:72px;font-weight:800;white-space:nowrap}"
                    ".arow{display:flex;align-items:center;gap:16px;margin-top:14px} .rob0{width:150px;height:150px} .mb0{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:6px 30px;font-size:80px;font-weight:700}</style>", 1)
assert HTML.count('id="mb2"') == 1 and "aura(R('mb2')" in HTML
dr.render(HTML, HERE, "words_told_pair", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
