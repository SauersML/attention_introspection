"""Method figure for attn_intro (Qwen3-32B, arithmetic task, instruction conditions). Real example: B.json id 400."""
import json, base64, subprocess, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.expanduser("~/attn_intro/qwen3-32b/B.json")))
ex = {x["cond"]: x for x in D if x["id"] == 400}
foc = ex["focus"]
assert all(ex[c]["ans"] == foc["ans"] for c in ex)
ANS = foc["ans"]                        # "97"
NTOK = foc["ntok"]                      # tokens per paragraph body
K = foc["k"] + 1                        # named paragraph (3)
body = foc["user"].split("\n\n")
paras = [p.split("] ", 1)[1] for p in body[1:5]]
q = body[5].split("\n")[1]              # "Unrelated question: what is 45 + 52? ..."
qs = {x["id"]: x for x in D if x["task"] == "arith" and x["cond"] == "focus"}
rows = []
for i in (400, 401, 402):
    u = qs[i]["user"]; s = u.split("what is ")[1].split("?")[0]
    rows.append((s, qs[i]["k"] + 1))
N_ARITH = len(qs)

robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
INK = "#1D1D1F"; AMB = "#F2B53A"; AMBD = "#C98A0B"; CELL = "#E6E6EB"
META = json.load(open(os.path.expanduser("~/attn_intro/qwen3-32b/meta.json")))
NL = META["NL"]; L0, L1 = META["LAYERS"][0], META["LAYERS"][-1]
assert META["LAYERS"] == list(range(L0, L1 + 1))
COND = [("mention P3", "#1A9E77", "Note that one of the paragraphs is labelled [P3]."),
        ("focus on P3", "#D2432F", "While you answer, focus your attention on [P3]."),
        ("do not attend to P3", "#7B3FE4", "While you answer, do not pay attention to [P3].")]
for _, _, line in COND:
    assert any(line in ex[c]["user"] for c in ex), line

def short(p, n=3):
    return " ".join(p.split()[:n]) + " …"

W = 2300
# ---------------- chat columns ----------------
cols = []
for name, colr, line in COND:
    plines = ""
    for i, p in enumerate(paras):
        cls = "k" if i + 1 == K else ""
        plines += f'<div class="pl"><b class="{cls}">[P{i+1}]</b> {html.escape(short(p, 2 if i == 0 else 3))}</div>'
    var = f'<div class="var" style="color:{colr}">{html.escape(line)}</div>'
    cols.append(f'''
<div class="col">
  <div class="cname" style="color:{colr}">{name}</div>
  <div class="ub">
    <div>Here are four paragraphs.</div>
    {plines}
    {var}
    <div>{html.escape(q).replace(" + ","&nbsp;+&nbsp;")}</div>
  </div>
  <div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{ANS}</div></div>
</div>''')

# ---------------- measurement panel (SVG) ----------------
sv = []
X0 = 40; Y0 = 20
LABW = 110; CW = 9; CG = 3; PITCH = CW + CG
rowH = 74; chipH = 58
first_words = [p.split()[:3] for p in paras]
def chipw(t): return 22 * len(t) + 34
y = Y0
rowsY = []
def text(x, yy, t, size=44, w=600, fill=INK, anchor="start", fam="Helvetica Neue, Helvetica, Arial"):
    sv.append(f'<text x="{x}" y="{yy}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-family="{fam}">{html.escape(t)}</text>')
right_ends = []
for i in range(4):
    hot = (i + 1 == K)
    yy = y + i * (rowH + 22)
    rowsY.append(yy)
    text(X0, yy + 46, f"P{i+1}", 46, 800, AMBD if hot else INK)
    x = X0 + LABW
    for wd in first_words[i]:
        cw = chipw(wd)
        sv.append(f'<rect x="{x}" y="{yy+8}" width="{cw}" height="{chipH}" rx="12" fill="{AMB if hot else CELL}"/>')
        text(x + cw / 2, yy + 51, wd, 41, 500, INK, "middle")
        x += cw + 6
    for c in range(NTOK[i] - 3):
        sv.append(f'<rect x="{x}" y="{yy+8}" width="{CW}" height="{chipH}" rx="3" fill="{AMB if hot else CELL}"/>')
        x += PITCH
    right_ends.append(x)
ybot = rowsY[-1] + rowH
# answer chips, right of the strips, arrows back to each paragraph row
gx = max(right_ends) + 150
mid = (rowsY[1] + rowsY[2]) / 2 + 8 + chipH / 2
sv.insert(0, f'<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
             f'<marker id="aa" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3" markerHeight="3" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{AMBD}"/></marker></defs>')
for i in range(4):
    hot = (i + 1 == K)
    ty = rowsY[i] + 8 + chipH / 2
    tx = right_ends[i] + (26 if hot else 14)
    sy = mid + (i - 1.5) * 22
    d = f"M{gx},{sy} C{gx-80},{sy} {tx+70},{ty} {tx},{ty}"
    if hot:
        sv.append(f'<path d="{d}" fill="none" stroke="{AMBD}" stroke-width="13" marker-end="url(#aa)"/>')
    else:
        sv.append(f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="4" marker-end="url(#ah)"/>')
ax = gx + 8
for t in ANS:
    cw = 86
    sv.append(f'<rect x="{ax}" y="{mid-46}" width="{cw}" height="92" rx="18" fill="white" stroke="{INK}" stroke-width="3.5"/>')
    text(ax + cw / 2, mid + 20, t, 58, 700, INK, "middle")
    ax += cw + 12
sv.append(f'<image href="data:image/png;base64,{robot}" x="{gx+32}" y="{rowsY[0]-6}" width="130" height="130"/>')
text(gx + 97, rowsY[3] + 30, "answer", 44, 700, INK, "middle")
text(gx + 97, rowsY[3] + 76, "tokens", 44, 700, INK, "middle")
SVG_H = rowsY[3] + rowH + 20
svg1 = f'<svg width="{W-60}" height="{SVG_H}" viewBox="0 0 {W-60} {SVG_H}">' + "".join(sv) + "</svg>"

# layer strip (36 layers, 12-30 used)
lay = []
for l in range(NL):
    on = L0 <= l <= L1
    lay.append(f'<rect x="{l*13}" y="0" width="10" height="56" rx="3" fill="{AMB if on else CELL}"/>')
laysvg = f'<svg width="{NL*13}" height="56">' + "".join(lay) + "</svg>"

# ---------------- pairing panel: one question, three conditions, x N ----------------
PC = [("mentioned", COND[0][1]), ("focus your attention", COND[1][1]), ("do not pay attention", COND[2][1])]
pv = []
CX0 = 240; CWD = 440; GAP = 280; CTOP = 110; CHT = 210; NSTK = 3; OFF = 20
cx = [CX0 + i * (CWD + GAP) for i in range(3)]
pv.append(f'<defs><marker id="cm" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="4" markerHeight="4" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>')
def ptext(x, yy, t, size=46, w=800, fill=INK, anchor="start"):
    pv.append(f'<text x="{x}" y="{yy}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" font-family="Helvetica Neue, Helvetica, Arial">{html.escape(t)}</text>')
for i, (nm, c) in enumerate(PC):
    x = cx[i]
    ptext(x + 4, CTOP - 26 - NSTK * OFF + 6, nm, 46, 800, c)
    for s_ in range(NSTK, 0, -1):   # grey copies behind = other questions
        pv.append(f'<rect x="{x + s_*OFF}" y="{CTOP - s_*OFF}" width="{CWD}" height="{CHT}" rx="20" fill="#F4F4F6" stroke="#C9C9D0" stroke-width="3"/>')
    pv.append(f'<rect x="{x}" y="{CTOP}" width="{CWD}" height="{CHT}" rx="20" fill="white" stroke="{c}" stroke-width="6"/>')
    for r in range(4):
        yy = CTOP + 24 + r * 42
        hot = (r + 1 == K)
        for j in range(24):
            pv.append(f'<rect x="{x + 26 + j*16.5}" y="{yy}" width="11" height="32" rx="3" fill="{AMB if hot else CELL}"/>')
ptext(CX0 - 30, CTOP + CHT / 2 + 16, q.split("what is ")[1].split("?")[0], 54, 800, INK, "end")
# compare: mentioned -> focus (straight), mentioned -> do not (arc under the cards)
yA = CTOP + CHT / 2
pv.append(f'<path d="M{cx[0]+CWD+NSTK*OFF+12},{yA} H{cx[1]-14}" stroke="{INK}" stroke-width="6" marker-end="url(#cm)"/>')
ptext((cx[0] + CWD + NSTK*OFF + cx[1]) / 2, yA - 24, "compare", 42, 800, INK, "middle")
yb = CTOP + CHT + 14
x1 = cx[0] + CWD / 2; x2 = cx[2] + CWD / 2; dip = yb + 120
pv.append(f'<path d="M{x1},{yb} C{x1},{dip} {x1+80},{dip} {x1+200},{dip} H{x2-200} C{x2-80},{dip} {x2},{dip} {x2},{yb+8}" fill="none" stroke="{INK}" stroke-width="6" marker-end="url(#cm)"/>')
pv.append(f'<rect x="{(x1+x2)/2-120}" y="{dip-34}" width="240" height="60" fill="white"/>')
ptext((x1 + x2) / 2, dip + 14, "compare", 42, 800, INK, "middle")
PH = dip + 40
svg2 = f'<svg width="{W-60}" height="{PH}" viewBox="0 0 {W-60} {PH}">' + "".join(pv) + "</svg>"
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:30px 30px 34px}}
.h{{font-size:52px;font-weight:800;line-height:1.15;margin:0 0 26px 6px}}
.cols{{display:flex;gap:28px}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:46px;font-weight:800;margin:0 0 14px 4px;white-space:nowrap}}
.ub{{flex:1;background:#1F8BFF;color:#fff;border-radius:34px;padding:24px 26px;font-size:40px;line-height:1.24;display:flex;flex-direction:column;gap:10px}}
.pl{{white-space:nowrap;overflow:hidden}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.var.none{{display:flex;align-items:center;background:transparent;border:4px dashed rgba(255,255,255,.85);color:#fff;font-weight:700}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:20px}}
.rob{{width:96px;height:96px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:56px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.sep{{height:4px;background:#EDEDF0;margin:38px 0 30px;border-radius:2px}}
.mhead{{display:flex;align-items:center;gap:30px;margin:0 0 18px 6px}}
.lay{{display:flex;align-items:center;gap:20px;font-size:44px;font-weight:700;margin:6px 0 0 6px}}
.tbl{{margin-top:8px}}
.tr,.thr{{display:flex;gap:10px;align-items:center;margin-bottom:16px}}
.rl{{width:200px;font-size:48px;font-weight:700;text-align:right;padding-right:6px}}
.th{{flex:1;font-size:46px;font-weight:800;padding-left:22px}}
.cw{{width:200px;text-align:center;font-size:42px;font-weight:800;}}
.cmp{{width:200px;display:flex;justify-content:center}}
.cn{{font-size:42px;font-weight:800;white-space:nowrap}}
.cell{{flex:1;height:96px;border:5px solid;border-radius:20px;display:flex;align-items:center;gap:14px;padding:0 18px;overflow:hidden}}
.rep{{display:flex;align-items:baseline;gap:50px;margin:18px 0 0 6px}}
.big{{font-size:84px;font-weight:800}}
.rnd{{font-size:46px;font-weight:800}}
.pk{{font-size:44px;font-weight:800;color:{AMBD}}}
</style></head><body>
<div class="h">Qwen3-32B gets the same message under three conditions; only one line differs</div>
<div class="cols">{"".join(cols)}</div>
<div class="sep"></div>
<div class="h">We measure how much of the answer's attention lands on P3</div>
{svg1}
<div class="lay">{laysvg}<span>layers {L0}–{L1} of {NL}, averaged over all heads</span></div>
<div class="sep"></div>
<div class="h">The same question's attention is compared across conditions</div>
{svg2}
<div class="rep"><span class="big">× {N_ARITH} questions</span><span class="rnd">The named paragraph is picked at random for each question.</span></div>
</body></html>'''
open(os.path.join(HERE, "method.html"), "w").write(HTML)
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
out = os.path.join(HERE, "method_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3000",
                "--force-device-scale-factor=1", f"--screenshot={out}", "file://" + os.path.join(HERE, "method.html")],
               check=True, capture_output=True)
print("ok", rows, N_ARITH)
from PIL import Image
im = Image.open(out).convert("RGB")
import numpy as np
a = np.asarray(im); nz = np.where((a < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 34)); im.save(os.path.join(HERE, "method.png")); print(im.size, im.size[1] / im.size[0])
