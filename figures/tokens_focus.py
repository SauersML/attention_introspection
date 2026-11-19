"""One prompt, every token of the named paragraph (Qwen3-32B, layer 28): % change in attention under
"Focus your attention on paragraph [P2]." vs the same prompt with "Note that one of the paragraphs is labelled [P2]." instead.
Data: results/qwen3-32b-grid-ci/G.json (grid_ci_pod.py), context 6 = the first "typical question" of grid_focus_L28
(the context just below the pool mean of the four-paragraph average; that average is +248% for this context).
Per token: tok_L28 = attention weights at layer 28, head mean, averaged over the answer-token queries (teacher-forced on the
no-instruction answer). Headline = span_L28 ratio for the named paragraph (mean over its measured tokens), focus2 / mention2.
The paragraph's first word shares a token with the space after the label and its final full stop shares one with the
newline after it: both are outside the measured span and drawn untinted. Colours: pal.pct_colour (shared scale, x1/4 .. x4)."""
import json, html, os, sys
import numpy as np
from matplotlib import colors
FIG = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, FIG)
from pal import PURPLE, INK, pct_colour
from hero_cols import CSS, AMB, BLUE, ROBOT, pct, render, MINUS
CID, K = 6, 2
R = json.load(open(os.path.join(FIG, "..", "results", "qwen3-32b-grid-ci", "G.json")))
r = next(x for x in R if x["id"] == CID)
F, M = r["vers"][f"focus{K}"], r["vers"][f"mention{K}"]
LINE = f"Focus your attention on paragraph [P{K}]."; BASE = f"Note that one of the paragraphs is labelled [P{K}]."
assert LINE in F["user"] and BASE in M["user"] and F["user"].replace(LINE, BASE) == M["user"]
a = np.array(F["tok_L28"][K - 1]); b = np.array(M["tok_L28"][K - 1]); toks = r["tokens"][K - 1]
assert len(a) == len(b) == len(toks)
assert abs(a.mean() / F["span_L28"][K - 1] - 1) < 1e-4 and abs(b.mean() / M["span_L28"][K - 1] - 1) < 1e-4
HEAD = 100 * (F["span_L28"][K - 1] / M["span_L28"][K - 1] - 1)
four = np.mean([100 * (r["vers"][f"focus{k}"]["span_L28"][k - 1] / r["vers"][f"mention{k}"]["span_L28"][k - 1] - 1) for k in range(1, 5)])
ANS = F["free"]; assert F["free_same"] and ANS.strip() == str(int(r["x"]) + int(r["y"]))
ratio = a / b
print(f"context {CID}: focus on P{K} {HEAD:+.2f}% (four-paragraph average {four:+.2f}%); answer {ANS}; "
      f"per-token ratio {ratio.min():.2f}..{ratio.max():.2f}, median {np.median(ratio):.2f}, {np.mean(ratio > 1) * 100:.0f}% of tokens up; "
      f"{np.mean(np.abs(np.log2(ratio)) > 2) * 100:.0f}% beyond the colour range")
para = r["paras"][K - 1]; body = "".join(toks)
first, rest = para.split(" ", 1)
assert para.startswith(first + body) or (first + body) in para, (first, body[:40])
tail = para[len(first) + len(body):]
print("untinted:", repr(first), repr(tail))

def chip(t, v):
    c = pct_colour(v); bg = colors.to_hex(c)
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    fg = "#FFFFFF" if lum < 0.55 else INK
    sp = " s" if t[:1].isspace() else ""
    return f'<span class="t{sp}" style="background:{bg};color:{fg}">{html.escape(t.strip())}</span>'
words = [[f'<span class="t pl">{html.escape(first)}</span>']]   # tokens grouped into words (a word starts at a token with a leading space) so a line never breaks inside a word
for t, v in zip(toks, ratio):
    if t[:1].isspace() and words[-1]: words.append([])
    words[-1].append(chip(t, v))
if tail.strip(): words[-1].append(f'<span class="t pl">{html.escape(tail.strip())}</span>')
chips = "".join(f'<span class="w">{"".join(w)}</span>' for w in words)
others = "".join(f'<div class="pl1"><b>[P{i + 1}]</b> {html.escape(" ".join(p.split()[:5]))} …</div>' for i, p in enumerate(r["paras"]) if i + 1 != K)
o = others.split("</div>")
pls = o[0] + "</div>" + f'<div class="pr"><b class="plab k">[P{K}]</b><div class="pp">{chips}</div></div>' + "</div>".join(o[1:])
W, BW = 2300, 860
stops = ",".join(f"{colors.to_hex(pct_colour(2.0 ** (-2 + 4 * i / 40)))} {i * 2.5}%" for i in range(41))
ticks = "".join(f'<span class="tk" style="left:{BW * (x + 2) / 4:.0f}px">{pct(100 * (2.0 ** x - 1)) if x else "0%"}</span>' for x in (-2, -1, 0, 1, 2))
page = f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}
body{{width:{W}px;padding:26px 30px 10px}}
.ub{{font-size:50px;gap:10px;padding:26px 30px}}
.pl1{{white-space:nowrap}}
.pr{{display:flex;align-items:flex-start;gap:14px}}
.plab{{flex:0 0 auto;line-height:1.2;margin-top:12px}}
.pp{{flex:1;background:#fff;border-radius:18px;padding:14px 16px 16px 10px;display:flex;flex-wrap:wrap;row-gap:14px;border:6px solid {AMB}}}
.t{{font-size:48px;line-height:1;padding:9px 3px 8px;border-radius:5px;margin-left:2px}}
.w{{display:inline-flex;margin-right:16px}} .w .t:first-child{{margin-left:0}} .pl{{color:{INK}}}
.var{{color:{PURPLE};font-size:54px}}
.bot{{display:flex;align-items:flex-start;justify-content:space-between;margin-top:26px;gap:40px}}
.hl{{display:flex;flex-direction:column;gap:12px}}
.big{{font-size:150px;font-weight:800;color:{PURPLE};letter-spacing:-2px;line-height:1}}
.ht{{font-size:52px;font-weight:700;line-height:1.2}}
.cmp{{font-size:50px;font-weight:700;margin-top:14px}}
.cub{{display:inline-block;background:{BLUE};border-radius:26px;padding:12px 14px;margin-top:10px}}
.cub .var{{color:{INK};font-size:48px}}
.key{{width:{BW}px;margin:70px 0 0 60px}}
.lc{{display:flex;flex-direction:column;flex:0 0 1000px}}
.kt{{font-size:50px;font-weight:700;margin:0 -60px 14px;text-align:center;white-space:nowrap}}
.kb{{height:64px;border-radius:12px;background:linear-gradient(to right,{stops})}}
.ks{{position:relative;height:70px}} .tk{{position:absolute;top:10px;transform:translateX(-50%);font-size:50px;font-weight:800}}
</style></head><body>
<div class="ub"><div>Here are four paragraphs.</div>{pls}
<div class="var">{html.escape(LINE)}</div>
<div>Unrelated question: what is {r["x"]}&nbsp;+&nbsp;{r["y"]}? Answer with just the number.</div></div>
<div class="bot">
 <div class="lc"><div class="mrow" style="margin-top:0"><img class="rob" src="data:image/png;base64,{ROBOT}"><div class="mb">{html.escape(ANS)}</div></div>
 <div class="key"><div class="kt">change in attention on each token</div><div class="kb"></div><div class="ks">{ticks}</div></div></div>
 <div class="hl"><div><span class="big">{pct(HEAD)}</span></div>
  <div class="ht">attention from Qwen3-32B's answer to [P{K}], layer 28</div>
  <div class="cmp">compared to a line that only mentions [P{K}]:</div>
  <div><span class="cub"><span class="var">{html.escape(BASE)}</span></span></div></div>
</div>
</body></html>"""
render(page, W, FIG, "tokens_focus", os.environ.get("TMPDIR", "/tmp"))
