# Analysis of the told-injection run: python3 analyze.py OUTDIR recs_*.jsonl
import json, sys, re, math, glob, os
import numpy as np
out = sys.argv[1]
R = [json.loads(l) for f in sys.argv[2:] for l in open(f)]
R.sort(key=lambda r: r["id"])
D = [2.0, 5.0, 10.0, 20.0]
rng = np.random.default_rng(0)
B = 10000
def boot(x):  # x: per-context values -> mean, 95% CI over contexts
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return dict(mean=None, lo=None, hi=None, n=0)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), n=int(len(x)))
PAT = re.compile(r"^\s*(No|Yes, paragraph \[P([1-4])\])\W*$")
def parse(t):
    m = PAT.match(t)
    if m: return ("no", None, True) if m.group(1) == "No" else ("yes", int(m.group(2)) - 1, True)
    s = t.strip().lower()
    if s.startswith("no"): return ("no", None, False)
    if s.startswith("yes"):
        mm = re.search(r"\[?p([1-4])\]?", s)
        return ("yes", int(mm.group(1)) - 1 if mm else None, False)
    return ("other", None, False)
def pyes(t):  # P(Yes) renormalised over Yes/No
    return 1 / (1 + math.exp(t["lp_no"] - t["lp_yes"]))
res = dict(n_contexts=len(R), doses={})
none_yes = np.array([parse(r["none"]["text"])[0] == "yes" for r in R], float)
none_py = np.array([pyes(r["none"]) for r in R])
res["control"] = dict(false_alarm=boot(none_yes), p_yes=boot(none_py),
                      texts=sorted(set(r["none"]["text"] for r in R)),
                      yesno_mass=boot([math.exp(r["none"]["lp_yes"]) + math.exp(r["none"]["lp_no"]) for r in R]))
odd = []
for m in D:
    det, hit, hitgiven, dpy, ppar, dpar, massx, fmt, digm = [], [], [], [], [], [], [], [], []
    for r in R:
        dk, hk, hg, dp, pp, dd, mx, fm = [], [], [], [], [], [], [], []
        for k in range(4):
            t = r["boost"][f"{k}|{m}"]
            ans, which, ok = parse(t["text"])
            dk.append(ans == "yes"); fm.append(ok)
            hk.append(ans == "yes" and which == k)
            if ans == "yes": hg.append(which == k)
            dp.append(pyes(t) - pyes(r["none"]))
            pp.append(math.exp(t["lp_par"][k]))
            dd.append(math.exp(t["lp_par"][k]) - math.exp(r["none"]["lp_par"][k]))
            mx.append(t["mass"][k] / r["none"]["mass"][k])
            if not ok: odd.append(dict(id=r["id"], k=k, m=m, text=t["text"]))
            digm.append(t["digit_mass"])
        det.append(np.mean(dk) - none_yes[R.index(r)]); hit.append(np.mean(hk))
        hitgiven.append(np.mean(hg) if hg else np.nan); dpy.append(np.mean(dp)); ppar.append(np.mean(pp)); dpar.append(np.mean(dd))
        massx.append(np.mean(mx)); fmt.append(np.mean(fm))
    allyes = [parse(r["boost"][f"{k}|{m}"]["text"]) for r in R for k in range(4)]
    res["doses"][str(m)] = dict(
        detection_rate=boot([np.mean([parse(r["boost"][f"{k}|{m}"]["text"])[0] == "yes" for k in range(4)]) for r in R]),
        detection_minus_false_alarm=boot(det),
        correct_and_detected=boot(hit),
        correct_given_detected=boot(hitgiven), n_detected_trials=int(sum(a[0] == "yes" for a in allyes)),
        p_yes_change=boot(dpy),
        p_boosted_paragraph_after_prefill=boot(ppar), p_boosted_change_vs_control=boot(dpar),
        attention_multiplier_on_question=boot(massx), wellformed=boot(fmt), digit_mass=boot(digm))
res["malformed_examples"] = odd[:40]; res["n_malformed"] = len(odd)
json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
f = lambda d: f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" if d["mean"] is not None else "n/a"
print("contexts", len(R), "control false alarm", f(res["control"]["false_alarm"]), "control P(yes)", f(res["control"]["p_yes"]))
print("control texts", res["control"]["texts"][:10])
for m in D:
    d = res["doses"][str(m)]
    print(f"x{m:g}: attn x{d['attention_multiplier_on_question']['mean']:.2f} | detect {f(d['detection_rate'])} det-FA {f(d['detection_minus_false_alarm'])} | "
          f"correct|det {f(d['correct_given_detected'])} (n={d['n_detected_trials']}) | dP(yes) {f(d['p_yes_change'])} | "
          f"P(boosted|prefill) {f(d['p_boosted_paragraph_after_prefill'])} change {f(d['p_boosted_change_vs_control'])} | wellformed {d['wellformed']['mean']:.3f}")
print("malformed", len(odd)); [print(o) for o in odd[:15]]
