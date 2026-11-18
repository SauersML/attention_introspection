# Token-level grid: Qwen3-32B, arithmetic task, 100 new contexts (seed 11) x 13 versions (grid_pod.py).
# Picks, for "focus" and for "ignore", the context just below and just above the pool's mean effect,
# and draws all four paragraphs' tokens coloured by % change vs the matching "only mentioned" run.
import json, html, subprocess, sys, numpy as np
from matplotlib import colors
G = sys.argv[1]
OUT = "/Users/user/connectome/islands/agents/resident-a/compute/fig/attn_intro/"
CHROME = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
sys.path.insert(0, OUT)
from pal import PURPLE, RED, INK, DIV
MINUS = "\u2212"; CLIP = 2.5  # colour scale in SD units within each 4x4 block
def pct(v): return f"+{v:.0f}%" if v >= 0 else f"{MINUS}{abs(v):.0f}%"
R = json.load(open(G)); byid = {r["id"]: r for r in R}

# ---- checks
rel = max(abs(np.mean(v["tok"][m]) / v["span_band"][m] - 1) for r in R for v in r["vers"].values() for m in range(4))
nfree = sum(v["free_same"] for r in R for k, v in r["vers"].items() if k != "none")
ntot = sum(1 for r in R for k in r["vers"] if k != "none")
acc = sum(r["ans_none"].strip() == str(r["x"] + r["y"]) for r in R)
print(f"max relative diff, paragraph mean from tokens vs span mean: {rel:.2e}")
print(f"free answers identical to no-instruction answer: {nfree}/{ntot}; no-instruction correct {acc}/{len(R)}")

# ---- effects and picks
def eff(r, cond): return float(np.mean([100 * (r["vers"][f"{cond}{k}"]["span_band"][k - 1] / r["vers"][f"mention{k}"]["span_band"][k - 1] - 1) for k in range(1, 5)]))
E = {c: {r["id"]: eff(r, c) for r in R} for c in ("focus", "ignore")}
MEAN = {c: float(np.mean(list(E[c].values()))) for c in E}
print("pool means", {c: round(MEAN[c], 2) for c in MEAN})
picks, used = [], set()
for c in ("focus", "ignore"):
    for side in ("below", "above"):
        cand = sorted((i for i in E[c] if (E[c][i] < MEAN[c] if side == "below" else E[c][i] > MEAN[c])), key=lambda i: abs(E[c][i] - MEAN[c]))
        i = next(i for i in cand if i not in used); used.add(i); picks.append((c, side, i))
        print(c, side, "context", i, f"effect {E[c][i]:+.2f}%")

# ---- tiles
def tile(t, v):
    u = (np.clip(v, -CLIP, CLIP) + CLIP) / (2 * CLIP); bg = colors.to_hex(DIV(u)); fg = "#FFFFFF" if abs(u - 0.5) > 0.32 else INK
    sp = t[:1].isspace()
    return f'<span class="t{" s" if sp else ""}" style="background:{bg};color:{fg}">{html.escape(t.strip()) or "&nbsp;"}</span>'
NAME = {"focus": "“Focus your attention on paragraph”", "ignore": "“Completely ignore paragraph”"}
allv = []
body = '<div class="grid"><div></div><div></div>' + "".join(f'<div class="hd">prompt names P{k}</div>' for k in range(1, 5))
for bi, (c, side, cid) in enumerate(picks):
    r = byid[cid]
    blk = np.concatenate([100 * (np.array(r["vers"][f"{c}{k}"]["tok"][m]) / np.array(r["vers"][f"mention{k}"]["tok"][m]) - 1) for m in range(4) for k in range(1, 5)])
    BSD = blk.std()  # each 4x4 block scaled by its own spread (0 stays "no change")
    col = PURPLE if E[c][cid] > 0 else RED
    body += f'<div class="blk" style="grid-row:span 4"><div class="bq">{NAME[c]}</div><div class="be" style="color:{col}">{pct(E[c][cid])}</div><div class="bt">a typical question</div></div>'
    for m in range(4):
        toks = r["tokens"][m]; joined = "".join(toks); txt = r["paras"][m]
        i0 = txt.index(joined)  # the first word (merged with the label's space) and the final "." (merged with the line break) are outside the measured span: shown untinted
        first, last = txt[:i0].strip(), txt[i0 + len(joined):].strip()
        body += f'<div class="pl{" gap" if m == 0 and bi else ""}">P{m + 1}</div>'
        for k in range(1, 5):
            a = np.array(r["vers"][f"{c}{k}"]["tok"][m]); b = np.array(r["vers"][f"mention{k}"]["tok"][m])
            ch = (100 * (a / b - 1)) / BSD; allv.extend(ch.tolist())
            cell = (f'<span class="t w">{html.escape(first)}</span>' if first else "") + "".join(tile(t, v) for t, v in zip(toks, ch)) + (f'<span class="t w">{html.escape(last)}</span>' if last else "")
            body += f'<div class="cell{" on" if k == m + 1 else ""}{" gap" if m == 0 and bi else ""}">{cell}</div>'
body += "</div>"
allv = np.array(allv)
print(f"per-token change: {np.mean(np.abs(allv) > CLIP) * 100:.1f}% of tiles beyond ±{CLIP} SD; 1st/99th pct {np.percentile(allv, 1):.1f} {np.percentile(allv, 99):.1f}")
stops = ",".join(f"{colors.to_hex(DIV(i / 20))} {i * 5}%" for i in range(21))
W = 5600
page = f"""<html><head><style>
body{{margin:0;background:#fff;font-family:Helvetica,Arial,sans-serif;color:{INK};width:{W}px}}
.wrap{{padding:30px 34px}}
.grid{{display:grid;grid-template-columns:430px 90px repeat(4,1fr);column-gap:18px;row-gap:12px}}
.hd{{font-size:64px;font-weight:bold;text-align:center;padding-bottom:6px}}
.blk{{display:flex;flex-direction:column;justify-content:center;padding-right:6px}}
.bq{{font-size:54px;font-weight:bold;line-height:1.12}}
.be{{font-size:110px;font-weight:bold;margin:14px 0 6px 0}}
.bt{{font-size:46px;font-weight:bold}}
.pl{{font-size:60px;font-weight:bold;display:flex;align-items:center;justify-content:center}}
.cell{{display:flex;flex-wrap:wrap;align-content:flex-start;row-gap:5px;font-size:27px;line-height:1;padding:8px;border:6px solid transparent;border-radius:10px}}
.cell.on{{border-color:{INK}}}
.gap{{margin-top:44px}}
.t{{padding:5px 2px;border-radius:3px}}
.s{{margin-left:7px}}
.w{{background:#fff;color:{INK}}}
.cbh{{display:flex;align-items:center;justify-content:center;margin-top:44px;font-size:64px;font-weight:bold}}
.hbar{{width:1500px;height:80px;border-radius:10px;margin:0 28px;background:linear-gradient(to right,{stops})}}
</style></head><body><div class="wrap">{body}
<div class="cbh"><span>{MINUS}{CLIP} SD</span><div class="hbar"></div><span>+{CLIP} SD</span><span style="margin-left:60px">change in attention vs just mentioning the paragraph name, scaled within each block</span></div>
</div></body></html>"""
open(OUT + "grid_ci.html", "w").write(page)
subprocess.run([CHROME, "--headless=new", "--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size={W},9000",
                "--screenshot=" + OUT + "grid_ci_raw.png", "file://" + OUT + "grid_ci.html"], check=True, capture_output=True)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
im = Image.open(OUT + "grid_ci_raw.png").convert("RGB"); px = np.asarray(im)
nz = np.where((px < 250).any(2).any(1))[0]; im = im.crop((0, 0, im.width, min(im.height, nz.max() + 30)))
im.save(OUT + "grid_ci.png"); im.save(OUT + "grid_ci.jpg", quality=90)
json.dump(dict(picks=[dict(cond=c, side=s, id=i, effect=E[c][i]) for c, s, i in picks], pool_mean=MEAN, effects=E,
               max_rel_diff=rel, free_same=nfree, free_total=ntot, none_correct=acc, n=len(R)),
          open("/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/grid/grid_ci_results.json", "w"), indent=1)
print("saved", OUT + "grid_ci.png", im.size)

# ---- 4x4 summary over the whole pool: rows = paragraph named, columns = paragraph measured
def gboot(lr, NB=10000, seed=0):
    rng = np.random.default_rng(seed); i = rng.integers(0, len(lr), (NB, len(lr))); s = lr[i].mean(1)
    return [100 * (np.exp(v) - 1) for v in (lr.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5))]
summ = dict(model="Qwen/Qwen3-32B", n=len(R), seed=11, statistic="geometric mean over contexts of attention(cond, named k, measured m) / attention(mention, named k, measured m), as % change; 95% bootstrap over contexts (10,000)",
            band_means={str(r["id"]): {v: r["vers"][v]["span_band"] for v in r["vers"]} for r in R}, matrix={})
for c in ("focus", "ignore"):
    M = [[gboot(np.log(np.array([r["vers"][f"{c}{k}"]["span_band"][m] / r["vers"][f"mention{k}"]["span_band"][m] for r in R]))) for m in range(4)] for k in range(1, 5)]
    summ["matrix"][c] = M
    print(c, "named \\ measured   P1 .. P4")
    for k in range(4): print(f"  P{k + 1}", "  ".join(f"{v[0]:+6.1f} [{v[1]:+.1f},{v[2]:+.1f}]" for v in M[k]))
json.dump(summ, open("/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/grid/grid_ci_summary.json", "w"))

# ---- per-context data for the hero violins: named paragraph, each k, vs mention and vs none
V = []
for r in R:
    nb = r["vers"]["none"]["span_band"]
    for k in range(1, 5):
        mb = r["vers"][f"mention{k}"]["span_band"]
        row = dict(id=r["id"], k=k, none=nb[k - 1], mention=mb[k - 1])
        for c in ("focus", "ignore"):
            a = r["vers"][f"{c}{k}"]["span_band"][k - 1]
            row[c] = a; row[f"{c}_vs_mention_pct"] = 100 * (a / mb[k - 1] - 1); row[f"{c}_vs_none_pct"] = 100 * (a / nb[k - 1] - 1)
        row["mention_vs_none_pct"] = 100 * (mb[k - 1] / nb[k - 1] - 1)
        V.append(row)
json.dump(dict(model="Qwen/Qwen3-32B", task="arith", seed=11, n_contexts=len(R), layers=list(range(21, 54)),
               wording={"focus": "Focus your attention on paragraph [Pk].", "ignore": "Completely ignore paragraph [Pk]. Do not look at it at all.",
                        "mention": "Note that one of the paragraphs is labelled [Pk].", "none": ""},
               measure="head-mean attention from answer-token queries (teacher-forced on the no-instruction answer) to the named paragraph's body tokens, per-token mean, mean over layers 21-53",
               rows=V), open("/Users/user/connectome/islands/agents/resident-a/compute/attn_intro/grid/violin_data_ci.json", "w"), indent=1)
for c in ("focus", "ignore"):
    for ref in ("mention", "none"):
        z = np.array([np.mean([v[f"{c}_vs_{ref}_pct"] for v in V if v["id"] == i]) for i in sorted({v["id"] for v in V})])
        print(f"{c} vs {ref}: per-context mean over k, mean {z.mean():+.1f}%, median {np.median(z):+.1f}%")
