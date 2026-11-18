# E1 follow-up (no new forwards): which heads carry the focus effect under each way of addressing the paragraph?
# Per head (layers 21-53): specific log ratio (target minus mean of other three, focus vs matched mention), mean over prompts.
# Correlate head profiles across addressing modes; attention mass on the instruction line itself; the six digit heads.
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from stats import *
R = torch.load(sys.argv[1]); out = {}
names = sorted({k.split("/")[0] for r in R for k in r["att"] if "/" in k})
SIX = [(20, 19), (20, 29), (23, 61), (25, 28), (25, 30), (26, 5)]
def perhead(r, key):
    a = r["att"][key].float()[:, :, :4] / torch.tensor(r["ntok"][key][:4], dtype=torch.float)
    return a  # [NL,H,4]
prof, prof_ctx = {}, {}
for nm in names:
    hs = []
    for r in R:
        k = r["k"]; lr = torch.log(perhead(r, f"{nm}/focus") + 1e-9) - torch.log(perhead(r, f"{nm}/mention") + 1e-9)
        sp = lr[..., k] - torch.cat([lr[..., j:j+1] for j in range(4) if j != k], -1).mean(-1)
        hs.append(sp[BAND])
    hs = torch.stack(hs)  # [N,33,64]
    prof[nm] = hs.mean(0).flatten().numpy(); prof_ctx[nm] = hs
C = np.corrcoef(np.stack([prof[n] for n in names]))
print("head-profile correlation (layers 21-53, 2112 heads)")
print(" " * 14 + " ".join(f"{n[:7]:>7s}" for n in names))
for i, n in enumerate(names): print(f"{n:14s}" + " ".join(f"{C[i,j]:7.2f}" for j in range(len(names))))
# split-half reliability of the label profile, as the ceiling for these correlations
N = len(R); h1 = prof_ctx["label"][: N // 2].mean(0).flatten(); h2 = prof_ctx["label"][N // 2 :].mean(0).flatten()
print("label split-half r", np.corrcoef(h1, h2)[0, 1])
for nm in names:
    a1 = prof_ctx[nm][: N // 2].mean(0).flatten(); a2 = prof_ctx[nm][N // 2 :].mean(0).flatten()
    out[f"splithalf/{nm}"] = float(np.corrcoef(a1, a2)[0, 1])
    out[f"corr_label/{nm}"] = float(np.corrcoef(prof[nm], prof["label"])[0, 1])
# share of total specific effect coming from the top-50 label heads
top = np.argsort(-prof["label"])[:50]
for nm in names:
    out[f"top50label_share/{nm}"] = float(prof[nm][top].sum() / np.clip(prof[nm], 0, None).sum())
# instruction-line attention (sum over its tokens), band mean, focus vs mention, and six digit heads
for nm in names:
    fi = [r["att"][f"{nm}/focus"].float()[BAND, :, 4].mean().item() for r in R]
    mi = [r["att"][f"{nm}/mention"].float()[BAND, :, 4].mean().item() for r in R]
    six_f = [np.mean([r["att"][f"{nm}/focus"].float()[L, h, 4].item() for L, h in SIX]) for r in R]
    six_m = [np.mean([r["att"][f"{nm}/mention"].float()[L, h, 4].item() for L, h in SIX]) for r in R]
    out[f"instr/{nm}"] = dict(focus=float(np.mean(fi)), mention=float(np.mean(mi)), six_focus=float(np.mean(six_f)), six_mention=float(np.mean(six_m)))
for k, v in out.items(): print(k, v if not isinstance(v, float) else round(v, 3))
json.dump(dict(out=out, C=C.tolist(), names=names), open(sys.argv[2], "w"), indent=1)
np.save(sys.argv[2].replace(".json", "_prof.npy"), np.stack([prof[n] for n in names]))
