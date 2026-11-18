import json, os, sys
import numpy as np
D = sys.argv[1]
meta = json.load(open(os.path.join(D, "meta.json"))); LAY = meta["LAYERS"]
R = json.load(open(os.path.join(D, "R.json")))
rng = np.random.default_rng(0); NB = 10000
n = len(R)
def band(att): return np.array(att)[LAY].mean(0)
out = {}
for which in ["att_mean", "att_first"]:
    A0 = np.array([band(r["none"][which]) for r in R])                      # [n,4] measured paragraph
    AI = np.array([[band(r["instr"][str(k)][which]) for k in range(4)] for r in R])  # [n,4 instructed,4 measured]
    LR = np.log(AI) - np.log(A0)[:, None, :]                                  # per-context log ratio
    idx = rng.integers(0, n, (NB, n))
    gm = np.exp(LR.mean(0)); bs = np.exp(LR[idx].mean(1))                  # [NB,4,4]
    lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    sd0 = A0.std(0, ddof=1)
    DS = (AI - A0[:, None, :]) / sd0[None, None, :]
    dsm = DS.mean(0); bsd = DS[idx].mean(1)
    dlo, dhi = np.percentile(bsd, 2.5, 0), np.percentile(bsd, 97.5, 0)
    diag_max_row = [bool(np.argmax(gm[k]) == k) for k in range(4)]
    diag_max_row_sd = [bool(np.argmax(dsm[k]) == k) for k in range(4)]
    # diagonal vs best off-diagonal in each row, paired bootstrap
    margin = []
    for k in range(4):
        off = [m for m in range(4) if m != k]
        d = LR[:, k, k][None] if False else None
        bm = LR[idx][:, :, k, :].mean(1)  # [NB,4]
        mg = bm[:, k] - bm[:, off].max(1)
        pt = LR[:, k, k].mean() - LR[:, k, off].mean(0).max()
        margin.append([float(np.exp(pt)), float(np.exp(np.percentile(mg, 2.5))), float(np.exp(np.percentile(mg, 97.5)))])
    # per-context: fraction of the 4 instructions where instructed paragraph got the biggest boost (log ratio) among the 4
    win = np.array([[np.argmax(LR[i, k]) == k for k in range(4)] for i in range(n)])  # [n,4]
    frac = win.mean(1)
    winsd = np.array([[np.argmax(DS[i, k]) == k for k in range(4)] for i in range(n)]).mean(1)
    bf = frac[idx].mean(1)
    out[which] = dict(ratio=gm.tolist(), ratio_lo=lo.tolist(), ratio_hi=hi.tolist(),
                      sd=dsm.tolist(), sd_lo=dlo.tolist(), sd_hi=dhi.tolist(), sd_none=sd0.tolist(), mean_none=A0.mean(0).tolist(),
                      diag_max_row=diag_max_row, diag_max_row_sd=diag_max_row_sd, diag_over_best_off=margin,
                      frac_mean=float(frac.mean()), frac_ci=[float(np.percentile(bf, 2.5)), float(np.percentile(bf, 97.5))],
                      frac_hist={str(v): int((np.isclose(frac, v)).sum()) for v in [0, .25, .5, .75, 1]},
                      frac_by_instr=win.mean(0).tolist(), frac_sd_mean=float(winsd.mean()),
                      frac_hist_sd={str(v): int((np.isclose(winsd, v)).sum()) for v in [0, .25, .5, .75, 1]})
same = np.array([[r["instr"][str(k)]["free_same"] for k in range(4)] for r in R])
correct_none = np.mean([r["ans_none"].strip() == str(r["x"] + r["y"]) for r in R])
correct_free = np.mean([[r["instr"][str(k)]["ans_free"].strip() == str(r["x"] + r["y"]) for k in range(4)] for r in R])
out["free_answer_same_as_none"] = dict(overall=float(same.mean()), by_instr=same.mean(0).tolist(), n_differ=int((~same).sum()), of=int(same.size),
                                       examples=[(r["ans_none"], r["instr"][str(k)]["ans_free"]) for r in R for k in range(4) if not r["instr"][str(k)]["free_same"]][:10])
out["correct_none"] = float(correct_none); out["correct_free_instr"] = float(correct_free)
out["n"] = n; out["ans_tokens"] = sorted(set(len(r["ans_none_ids"]) for r in R))
json.dump(out, open(os.path.join(D, "rotate_results.json"), "w"), indent=1)
for which in ["att_mean", "att_first"]:
    o = out[which]
    print(f"\n=== {which}: ratio to no instruction (rows instructed P1..P4, cols measured) ===")
    for k in range(4):
        print(f"P{k+1}: " + "  ".join(f"{o['ratio'][k][m]:.3f} [{o['ratio_lo'][k][m]:.3f},{o['ratio_hi'][k][m]:.3f}]" for m in range(4)))
    print("SD units:")
    for k in range(4):
        print(f"P{k+1}: " + "  ".join(f"{o['sd'][k][m]:+.3f} [{o['sd_lo'][k][m]:+.3f},{o['sd_hi'][k][m]:+.3f}]" for m in range(4)))
    print("diag max per row (ratio):", o["diag_max_row"], " (SD):", o["diag_max_row_sd"])
    print("diag / best off-diag ratio per row:", o["diag_over_best_off"])
    print("per-context fraction:", o["frac_mean"], o["frac_ci"], o["frac_hist"], "by instr", o["frac_by_instr"], "| SD-based:", o["frac_sd_mean"], o["frac_hist_sd"])
print("\nfree answers same as none:", out["free_answer_same_as_none"]); print("correct none", correct_none, "instr", correct_free, "ans token counts", out["ans_tokens"])
