# E1: does instructed focus need the [Pk] label? Address the target paragraph by label, ordinal, topic, quote,
# and the label instruction in Chinese, French, base64, emoji, a couplet. Each focus line has its own matched
# "mention" line that refers to the paragraph the same way but asks for nothing. Answer teacher-forced (no-instruction greedy answer).
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from xlib import *

N = int(sys.argv[1]); OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
tok, model = load()
ctxs = contexts(tok, N)

def first_words(t, n=5): return " ".join(t.split()[:n])

def conds(c):
    k = c["k"]; K = k + 1; title = c["paras"][k]["title"]; fw = first_words(c["paras"][k]["text"])
    L = dict(
      label=(True, f"Focus your attention on paragraph [P{K}].", f"Note that one of the paragraphs is labelled [P{K}]."),
      ordinal_nolab=(False, f"Focus your attention on the {ORD[k]} paragraph.", f"Note that there is a {ORD[k]} paragraph."),
      ordinal_lab=(True, f"Focus your attention on the {ORD[k]} paragraph.", f"Note that there is a {ORD[k]} paragraph."),
      topic_nolab=(False, f"Focus your attention on the paragraph about {title}.", f"Note that one of the paragraphs is about {title}."),
      quote_nolab=(False, f'Focus your attention on the paragraph that begins "{fw}".', f'Note that one of the paragraphs begins "{fw}".'),
      chinese=(True, f"请把注意力集中在段落[P{K}]上。", f"请注意，其中一个段落标为[P{K}]。"),
      french=(True, f"Concentre ton attention sur le paragraphe [P{K}].", f"Note qu'un des paragraphes est étiqueté [P{K}]."),
      base64=(True, b64(f"Focus your attention on paragraph [P{K}]."), b64(f"Note that one of the paragraphs is labelled [P{K}].")),
      emoji=(True, f"👀👉 [P{K}]", f"🏷️ [P{K}]"),
      couplet=(True, f"Of all four passages laid out for you,\nlet [P{K}] hold your gaze the whole way through.",
                     f"Of all four passages laid out for you,\none carries [P{K}] as its label too."),
    )
    return L

R = []
for n, c in enumerate(ctxs):
    rec = dict(id=c["id"], k=c["k"], att={}, lp={}, ntok={})
    base = {}
    for lab in (True, False):
        d = para_doc(c, labels=lab)
        ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
        base[lab] = (ids, sp)
    ans = generate(tok, model, base[True][0])
    rec["ans"] = tok.decode(ans)
    for lab in (True, False):
        ids, sp = base[lab]
        spans = [sp[f"p{i}"] for i in range(4)]
        a, lp = measure(model, ids, ans, spans); key = "none_lab" if lab else "none_nolab"
        rec["att"][key] = a; rec["lp"][key] = lp
        rec["ntok"][key] = [b - a_ for a_, b in spans]
    for name, (lab, foc, men) in conds(c).items():
        for kind, line in (("focus", foc), ("mention", men)):
            d = para_doc(c, labels=lab, instr=line)
            ids, sp = encode(tok, [("user", d.s)], {0: d.sp})
            spans = [sp[f"p{i}"] for i in range(4)] + [sp["instr"]]
            a, lp = measure(model, ids, ans, spans)
            rec["att"][f"{name}/{kind}"] = a; rec["lp"][f"{name}/{kind}"] = lp
            rec["ntok"][f"{name}/{kind}"] = [b - a_ for a_, b in spans]
    R.append(rec)
    if n % 10 == 0: log("ctx", n, rec["ans"])
torch.save(R, f"{OUT}/e1.pt"); log("done")
