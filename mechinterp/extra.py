# Extra tables from sweep.pt for figures: per-layer head-mean change (with bootstrap CI), attention to the instruction digit.
import json, torch
d = torch.load("/workspace/mi/out/sweep.pt", weights_only=False)
att = d["att"].float(); nC = att.shape[0] // 4; NL, H = att.shape[1], att.shape[3]
att = att.view(nC, 4, NL, 4, H, -1)  # ctx cond L k H span
C = {"none": 0, "mention": 1, "focus": 2, "ignore": 3}
T = torch.stack([att[:, :, :, k, :, k] for k in range(4)], 3)  # ctx cond L k H : attention per token to named paragraph
Tm = T.mean(3)  # ctx cond L H
g = torch.Generator().manual_seed(0)
def boot(x, nb=2000):
    idx = torch.randint(0, len(x), (nb, len(x)), generator=g); m = x[idx].mean(1)
    return [x.mean().item(), m.quantile(.025).item(), m.quantile(.975).item()]
out = {"nC": nC}
for cond in ["focus", "ignore", "none"]:
    hm = Tm[:, C[cond]].mean(-1); hb = Tm[:, C["mention"]].mean(-1)  # ctx L
    out[f"layer_rel/{cond}"] = [boot(hm[:, L] / hb[:, L] * 100 - 100) for L in range(NL)]
# digit attention (span 9) from answer tokens, per head, per condition
dig = att[..., 9].mean(3)  # ctx cond L H (mean over k)
out["digit_att_mean"] = {c: dig[:, i].mean(0).tolist() for c, i in C.items() if c != "none"}
MOV = [(20, 19), (20, 29), (23, 61), (25, 28), (25, 30), (26, 5)]
out["digit_att_movers"] = {c: [boot(dig[:, C[c], L, h]) for L, h in MOV] for c in ["mention", "focus", "ignore"]}
out["digit_att_layer_headmean"] = {c: dig[:, C[c]].mean((0, 2)).tolist() for c in ["mention", "focus", "ignore"]}
# instruction-span attention (span 8)
ins = att[..., 8].mean(3)
out["instr_att_movers"] = {c: [boot(ins[:, C[c], L, h]) for L, h in MOV] for c in ["mention", "focus", "ignore"]}
json.dump(out, open("/workspace/mi/out/extra.json", "w"))
print("ok", nC, [round(v[0], 1) for v in out["layer_rel/focus"][20:35]])
print("movers digit focus", [round(v[0], 3) for v in out["digit_att_movers"]["focus"]], "mention", [round(v[0], 3) for v in out["digit_att_movers"]["mention"]])
