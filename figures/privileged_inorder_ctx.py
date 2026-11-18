# Context file for the in-order rerun (results/privileged_inorder): the same contexts as privileged_ctx.json with lab and opt in order
# and the user text the run recorded (prompt_user). Checked per prompt: lab, opt, ntok match the record; paragraphs appear in order.
import json, os
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(os.path.dirname(H), "results/privileged_inorder/data/recs_0.jsonl")
ctx = {c["id"]: c for c in json.load(open(os.path.join(H, "privileged_ctx.json")))}
out = []
for r in map(json.loads, open(R)):
    c = dict(ctx[r["id"]]); assert c["ntok"] == r["ntok"] and r["lab"] == r["opt"] == [1, 2, 3, 4]
    u = r["prompt_user"]; pos = [u.index(f"[P{i + 1}] " + s) for i, s in enumerate(c["segs"])]; assert pos == sorted(pos)
    c.update(lab=r["lab"], opt=r["opt"], user=u); out.append(c)
json.dump(out, open(os.path.join(H, "privileged_inorder_ctx.json"), "w")); print(len(out))
