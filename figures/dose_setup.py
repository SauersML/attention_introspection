"""dose_setup: the real prompt (id 300, "focus your attention on" wording, paragraphs cut to three words) with the instruction line
as a white bubble, the words that change between wordings in a dashed box, the 0% reference line said under the reply, then dose.png."""
import json, base64, subprocess, os, html, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from pal import PURPLE, RED, INK
D = os.path.join(HERE, "..", "results", "qwen3-32b-dose") + "/"
meta = json.load(open(D + "meta.json")); R = json.load(open(D + "dose.json"))
ex = [r for r in R if r["id"] == 300 and r["cond"] == "3_focus"][0]
u = ex["user"]; K = ex["k"] + 1
line = meta["conds"]["3_focus"][0].format(k=K); assert line in u
ment = meta["conds"]["1_mention"][0].format(k=K)
VAR = "Focus your attention on"; assert line.startswith(VAR)
parts = u.split("\n\n"); paras = parts[1:5]; tail = "\n\n".join(parts[5:])
pre, post = tail.split(line)
AMB = "#F2B53A"; W = 2300
robot = base64.b64encode(open("/Users/user/watching/figures/assets/robot.png", "rb").read()).decode()
DOSE = base64.b64encode(open(os.path.join(HERE, "dose.png"), "rb").read()).decode()
def para(p):
    lab, txt = p.split("] ", 1); i = int(lab[2:]); txt = " ".join(txt.split()[:3]) + " …"
    return f'<div class="pl"><b class="{"k" if i == K else ""}">{html.escape(lab)}]</b> {html.escape(txt)}</div>'
HTML = f'''<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{width:{W}px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK};padding:20px 26px 10px}}
.ub{{background:#1F8BFF;color:#fff;border-radius:34px;padding:22px 28px;font-size:52px;line-height:1.22;display:flex;flex-direction:column;gap:12px}}
.k{{background:{AMB};color:{INK};border-radius:10px;padding:0 6px}}
.var{{background:#fff;color:{INK};border-radius:16px;padding:10px 16px;font-weight:800;font-size:52px}}
.dv{{border:5px dashed {INK};border-radius:12px;padding:0 8px}}
.vrow{{display:flex;align-items:center;gap:20px}} .lab{{font-size:48px;font-weight:800;color:#fff;white-space:nowrap}}
.ref{{font-size:50px;font-weight:700;margin-left:24px;line-height:1.15}}
.mrow{{display:flex;align-items:center;gap:16px;margin-top:12px}}
.rob{{width:100px;height:100px}}
.mb{{background:#fff;border:3px solid #D9D9DE;border-radius:30px;padding:10px 30px;font-size:62px;font-weight:700}}
.dose{{width:100%;display:block;margin-top:14px}}
</style></head><body>
<div class="ub"><div>{html.escape(parts[0].strip())}</div>{"".join(para(p) for p in paras)}
<div class="vrow"><div class="var"><span class="dv">{html.escape(VAR)}</span>{html.escape(line[len(VAR):])}</div><div class="lab">← boxed words: 7 wordings, below</div></div><div>{html.escape(post.strip())}</div></div>
<div class="mrow"><img class="rob" src="data:image/png;base64,{robot}"><div class="mb">{html.escape(ex["ans_free"].strip())}</div><div class="ref">0% = the same prompt with the line<br>“{html.escape(ment)}”</div></div>
<img class="dose" src="data:image/png;base64,{DOSE}"></body></html>'''
CH = "/Users/user/.pw/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
hp = os.path.join(HERE, "dose_setup.html"); open(hp, "w").write(HTML); raw = os.path.join(HERE, "dose_setup_raw.png")
subprocess.run(["nice", "-n", "19", CH, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},5000",
                "--force-device-scale-factor=1", f"--screenshot={raw}", "file://" + hp], check=True, capture_output=True)
im = Image.open(raw).convert("RGB"); ar = np.asarray(im); nz = np.where((ar < 250).any(2).any(1))[0]
im = im.crop((0, 0, W, nz[-1] + 20)); im.save(os.path.join(HERE, "dose_full.png")); im.save(os.path.join(HERE, "dose_full.jpg"), quality=92); print(im.size)
print(repr(pre[-80:]))
