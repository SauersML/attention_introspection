"""words_told_pair_lp: words_told_pair.py with the right panel in probabilities instead of top answers.
RIGHT here: P(boosted word's number) after the prefill "Yes, word [W" = softmax of the stored log-probs of digits 1-8 (renormalised over
those 8), per context the mean over the 8 boosted words, relative to no boost (exactly 12.5% by averaging over the 8 rotations).
words_told_pair: told_pair.py's design for the told-injection run on single-word lists (Qwen3-32B, 240 contexts; results/told_inject_words,
told_inject_words/told_inject_words.py). Each context: the researcher's intro, eight single-token nouns from eight categories, one per line,
labelled [W1]..[W8], and an arithmetic line; Qwen's real unboosted answer; then the second user turn "Trial 1: Do you notice your attention
being boosted onto one of the words? If so, which one?". The boost adds log(multiplier) to the pre-softmax attention scores toward the boosted
word's token from every query from "Trial 1" on, layers 21-53. Top: the method drawing from injb_draw.py (example context nearest the median of
P(boosted digit) at multiplier 10 among those whose top prefilled digit is the boosted word, word 3 boosted; Qwen's real greedy reply "No"
and the prefilled "Yes, word [W" with the real top digit, black arrows to the two panel headers).
LEFT: probability of "Yes" as the reply's first token, renormalised over Yes/No (exp(lp_yes) / (exp(lp_yes) + exp(lp_no)), stored log-probs),
per context the mean over the 8 boosted words, as relative change of the mean over contexts from no boost (the "none" record); whiskers 95%
bootstrap over contexts; p: sign-flip over contexts at the top dose; raw mean probabilities above each bar. Greedy replies starting with "Yes":
0 of 240 control and 0 of 7,680 boosted trials (240 contexts x 8 words x 4 doses), written on the panel.
RIGHT: top answer after the prefill "Yes, word [W" (boost on) = argmax of the stored log-probs of digits 1-8; per context, the percent of the
8 boosted words where the top answer is the boosted word's number, relative to no boost (exactly 12.5% for every context: one fixed top answer
matches 1 of 8 rotations); whiskers 95% bootstrap over contexts (20,000 resamples); p: two-sided sign-flip over contexts at the top dose
(200,000 flips). Dose labels: measured attention mass on the boosted word's token, boost / no boost, mean over contexts and words.
usage: OMP_NUM_THREADS=2 nice -n 19 python3 words_told_pair_lp.py [--keep]   (writes figures/words_told_pair_lp.png and .jpg)"""
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
assert NY == 0 and YC.sum() == 0, (NY, YC.sum())   # Qwen never says Yes greedily: written on the panel
PY = lambda t: 1 / (1 + np.exp(t["lp_no"] - t["lp_yes"]))   # P(Yes) renormalised over Yes/No
assert all(set(r["boost"]) == {f"{k}|{m}" for k in range(NW) for m in MS} for r in R)
PC = np.array([PY(r["none"]) for r in R]); PYB = {m: np.array([np.mean([PY(T(r, k, m)) for k in range(NW)]) for r in R]) for m in MS}
PNY = tc.signflip(PYB[20.0] - PC, NF)
RY = {m: (PYB[m].mean() / PC.mean() - 1, tc.boot(lambda ix, m=m: PYB[m][ix].mean() / PC[ix].mean() - 1, len(R))) for m in MS}
print("P(Yes) %:", f"{100*PC.mean():.2f} ->", [f"{100*PYB[m].mean():.2f}" for m in MS], "relative:", [f"{100*RY[m][0]:+.0f} [{100*RY[m][1][0]:+.0f}, {100*RY[m][1][1]:+.0f}]" for m in MS], "p", PNY)
# right: probability of the boosted word's number after the prefill, renormalised over the eight digit log-probs
def pr(lp, k): e = np.exp(np.asarray(lp) - max(lp)); return e[k] / e.sum()
AC = np.array([np.mean([pr(r["none"]["lp_par"], k) for k in range(NW)]) for r in R])
assert np.allclose(AC, 1 / NW)   # averaging over the eight rotations: the no-boost probabilities sum to 1
AB = {m: np.array([np.mean([pr(T(r, k, m)["lp_par"], k) for k in range(NW)]) for r in R]) for m in MS}
PN = tc.signflip(AB[20.0] - 1 / NW, NF)
RN = {m: (AB[m].mean() * NW - 1, tc.boot(lambda ix, m=m: AB[m][ix].mean() * NW - 1, len(R))) for m in MS}
print("Yes:", f"{NY} of {NT:,} boosted, {int(YC.sum())} of {len(R)} control")
print("attention x:", [f"{MASSES[m]:.2f}" for m in MS])
print("mean P(boosted) %:", [f"{100*AB[m].mean():.1f}" for m in MS], "relative:", [f"{100*RN[m][0]:+.0f} [{100*RN[m][1][0]:+.0f}, {100*RN[m][1][1]:+.0f}]" for m in MS], "p", PN)
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
VW = W - 2 * PAD; GAP = 60; LM = 370; VH = 1480; TOPH = 330; BOT = 280
PWL = 1700; PWR = VW - GAP - PWL   # the right panel wider: its big labels ("+388%") need wider columns than the left panel's "0%"
CWL, CWR = (PWL - LM) / 4, (PWR - LM) / 4
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK})
fig = plt.figure(figsize=(VW / 100, VH / 100), dpi=100)
def axis(lo, hi):   # full-width axes (x in canvas px), one per panel so each has its own y scale
    a = fig.add_axes([0, BOT / VH, 1, (VH - TOPH - BOT) / VH]); a.set_xlim(0, VW); a.set_ylim(lo, hi); a.patch.set_visible(False)
    a.set_xticks([]); a.set_yticks([]); [s_.set_visible(False) for s_ in a.spines.values()]; return a
L = dict(PW=PWL, LM=LM, CW=CWL, VW=VW, VH=VH, ylo=-0.034, yhi=0.34); Rg = dict(PW=PWR, LM=LM, CW=CWR, VW=VW, VH=VH, ylo=-0.34, yhi=3.4)
axL = axis(L["ylo"], L["yhi"]); axR = axis(Rg["ylo"], Rg["yhi"])
# left: P(Yes) renormalised over Yes/No, relative to no boost; the greedy count written on the panel
tc.bars_panel(fig, axL, 0, L, MS, RY, PC.mean(), {m: PYB[m].mean() for m in MS}, MASSES, (0.1, 0.2, 0.3), "word", PNY, NF,
              base_lab=lambda b: f"{100*b:.2f}", rate_lab=lambda r: f"{100*r:.2f}%")
axL.text(LM + 1.5 * CWL, 0.25, f"Qwen\u2019s reply starts\nwith \u201cYes\u201d: 0 of {NT:,}", ha="center", va="center", fontsize=dr.px(84), fontweight="bold", color=INK, linespacing=1.1)
tc.bars_panel(fig, axR, PWL + GAP, Rg, MS, RN, 1 / NW, {m: AB[m].mean() for m in MS}, MASSES, (1, 2, 3), "word", PN, NF,
              base_lab=lambda b: f"{100*b:.1f}%")
VP = os.path.join(HERE, "_v_words_told_pair_lp.png"); fig.savefig(VP, dpi=100, facecolor="white"); plt.close(fig)

# ---------------- page ----------------
bubble = dr.told_words_bubble(META, rec["words"], K, QLINE)
pre = f'<div class="arow" id="rowA"><img class="rob0" src="data:image/png;base64,ROBOT_PLAIN"><div class="mb0">{rec["ans"].strip()}</div></div>'
reply = f'<span class="part" id="ry">No</span>'
titles = (f'<div class="cols"><div class="ct" style="flex:none;width:{PWL - LM:.0f}px;margin-left:{LM}px"><span class="cn" id="c1">average probability<br>of \u201cYes\u201d</span></div>'
          f'<div class="ct" style="flex:none;width:{PWR - LM:.0f}px;margin-left:{GAP + LM:.0f}px"><span class="cn" id="c2">average probability Qwen gives<br>the boosted word\u2019s number</span></div></div>')
HTML = dr.page(W, bubble, "Do you notice your attention being boosted onto one of the words? If so, which one?", reply, titles, VP, VW, f"{MASSES[M]:.1f}",
               pre=pre, inj=f"boost attention<br>toward word {K + 1}", arrows=(("ry", "c1"), ("rp", "c2")), tipdrop=-2, inj_px=105, inj_near=(-18, -60))
mb2 = f'<div class="mb" id="mb2"><span class="pfb">Yes, word [W</span><span class="part" id="rp">{PICK}</span>]</div>'
HTML = HTML.replace('<span class="part" id="ry">No</span></div>', '<span class="part" id="ry">No</span></div><div class="orx">(or)</div>' + mb2, 1)   # Sauers: "(or)", not "No Yes"
HTML = HTML.replace("g+=aura(R('mb'),30);", "g+=aura(R('mb'),30); g+=aura(R('mb2'),30);", 1)
HTML = HTML.replace("</style>", ".ub{margin-right:700px;font-size:62px}#rq{margin-left:1500px;margin-right:500px;font-size:62px}.mrow{margin-left:1250px;gap:40px}.orx{font-size:50px;font-weight:800;color:"+INK+";align-self:center;white-space:nowrap}.mb{font-size:96px}.cn{font-size:86px}.cols{margin-top:250px}"
                    f".pfb{{border:5px dashed {INK};border-radius:14px;padding:0 10px;margin-right:6px;position:relative}}"
                    ".pfb::after{content:'prefilled';position:absolute;left:50%;transform:translateX(-50%);top:100%;margin-top:30px;font-size:72px;font-weight:800;white-space:nowrap}"
                    ".arow{display:flex;align-items:center;gap:16px;margin-top:14px} .rob0{width:150px;height:150px} .mb0{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:6px 30px;font-size:80px;font-weight:700}</style>", 1)
assert HTML.count('id="mb2"') == 1 and "aura(R('mb2')" in HTML
dr.render(HTML, HERE, "words_told_pair_lp", W, jpg=True, keep="--keep" in sys.argv)
if "--keep" not in sys.argv: os.remove(VP)
