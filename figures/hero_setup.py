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
COND = [("focus your attention", "#5E2B97", "While you answer, focus your attention on [P3]."),
        ("do not pay attention", "#C0392B", "While you answer, do not pay attention to [P3].")]
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

import base64 as _b
VIO = _b.b64encode(open(os.path.join(HERE, "vhero.png"), "rb").read()).decode()
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:30px 30px 20px}}
.cols{{display:flex;gap:40px;padding-left:330px}}
.col{{flex:1;min-width:0;display:flex;flex-direction:column}}
.cname{{font-size:58px;font-weight:800;margin:0 0 14px 4px;white-space:nowrap}}
.ub{{flex:1;background:#1F8BFF;color:#fff;border-radius:34px;padding:24px 26px;font-size:40px;line-height:1.24;display:flex;flex-direction:column;gap:10px}}
.pl{{white-space:nowrap;overflow:hidden}}
.var{{background:#fff;border-radius:16px;padding:10px 16px;font-weight:700;line-height:1.18}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:20px}}
.rob{{width:96px;height:96px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:12px 34px;font-size:56px;font-weight:700;box-shadow:0 3px 8px rgba(0,0,0,.08)}}
.vio{{width:100%;display:block;margin-top:10px}}
</style></head><body>
<div class="cols">{"".join(cols)}</div>
<img class="vio" src="data:image/png;base64,{VIO}">
</body></html>'''
open(os.path.join(HERE, "hero2.html"), "w").write(HTML)
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
out = os.path.join(HERE, "hero2_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},3000",
                "--force-device-scale-factor=1", f"--screenshot={out}", "file://" + os.path.join(HERE, "hero2.html")], check=True, capture_output=True)
from PIL import Image
import numpy as np
im = Image.open(out).convert("RGB"); a = np.asarray(im); nz = np.where((a < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, "hero.png")); im.save(os.path.join(HERE, "hero.jpg"), quality=92); print(im.size)
