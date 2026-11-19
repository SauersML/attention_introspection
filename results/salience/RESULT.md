# Salience vs attention (Qwen3-32B, 240 contexts)

**Misleading prompt, not used:** this prompt tells Qwen that only attention boosts happen, so the content-addition (B, B2) and random-direction (R) arms here misled Qwen; their data stay in data/ but are not used in any figure or claim. Only the honest arms (none, A, and the pick-at-random question QA) are used, in figures/salience.py.

Same told prompt as told_inject ("I am capable of artificially boosting your attention onto one of the four paragraphs below … Do you notice …? If so, which one?"). Arms, each paragraph k boosted in turn, layers 21–53, at the question and answer tokens:
- none: no boost.
- A: attention boost ×20 onto paragraph k (log 20 added to the pre-softmax scores, all heads).
- B: paragraph k's content direction (mean residual over paragraph k minus mean over all four paragraphs, from the unboosted run) added into the residual stream, sized per layer so the residual change matches A's.
- B2: 2× B.

Scorer output (`python3 salience/score.py`):

```
greedy_yes none 0.125 A 0.218 rel +74% [+33,+149] p 5e-05 B 0.215 rel +72% [+30,+148] p 5e-05 B2 0.173 rel +38% [+1,+104] p 0.0389 A-B p 0.91 A-B2 p 0.0161
p_yes none 0.324 A 0.340 rel +5% [-1,+12] p 0.0944 B 0.302 rel -7% [-11,-2] p 0.00525 B2 0.240 rel -26% [-31,-20] p 5e-05 A-B p 5e-05 A-B2 p 5e-05
top_is_k none 0.250 A 0.585 rel +134% [+123,+145] p 5e-05 B 0.352 rel +41% [+32,+49] p 5e-05 B2 0.388 rel +55% [+45,+65] p 5e-05 A-B p 5e-05 A-B2 p 5e-05
p_k none 0.250 A 0.535 rel +114% [+106,+121] p 5e-05 B 0.348 rel +39% [+35,+44] p 5e-05 B2 0.379 rel +52% [+46,+57] p 5e-05 A-B p 5e-05 A-B2 p 5e-05
dres L21,30,40,53,54: {'A': [0.0, 24.1, 31.4, 158.1, 186.0], 'B': [0.0, 33.3, 41.1, 192.1, 230.4], 'B2': [0.0, 67.5, 84.9, 361.4, 430.9]}
cos: {'A': [0.0, 0.02, -0.01, -0.0, 0.0], 'B': [0.0, 0.68, 0.65, 0.54, 0.53], 'B2': [0.0, 0.69, 0.66, 0.59, 0.58]}
c: [6.1, 8.5, 12.9, 42.9] resnorm [171, 214, 582] mass A/B 9.94088186221366 1.1543149147717613
{'A': [('No', 629), ('No.', 122), ('Yes, paragraph [P1]', 112), ('Yes, paragraph [P2]', 46), ('Yes, paragraph [P3]', 35), ('Yes, paragraph [P4]', 16)], 'B': [('No', 541), ('No.', 213), ('Yes, paragraph [P1]', 137), ('Yes, paragraph [P2]', 50), ('Yes, paragraph [P3]', 9), ('Yes, paragraph [P4].', 3), ('Yes, paragraph [P4]', 3), ('Yes, paragraph [P1].', 2)], 'B2': [('No', 426), ('No.', 365), ('Yes, paragraph [P1]', 73), ('Yes, paragraph [P2]', 25), ('Yes, paragraph [P1].', 20), ('Yes, paragraph [P2].', 7), ('Yes, paragraph [P3].', 6), ('Yes, paragraph [P3]', 6)], 'none': [('No', 204), ('Yes, paragraph [P1]', 18), ('Yes, paragraph [P2]', 10), ('No.', 6), ('Yes, paragraph [P3]', 2)]}
digit mass min 0.08248647270097256
```

Greedy reply starts "Yes": none 12.5%, A 21.8% (+74%), B 21.5% (+72%), B2 17.3% (+38%); A vs B p = 0.91.
Top digit after the prefill "Yes, paragraph [P" is the boosted paragraph: none 25%, A 58.5% (+134%), B 35.2% (+41%), B2 38.8% (+55%); A vs B p < 10⁻⁴ (sign-flip floor).

Geometry: A's residual change is almost orthogonal to the content direction (cos ≈ 0 at every band layer), while B's is aligned with it (cos .53–.68). B's residual change is slightly larger than A's (e.g. 33 vs 24 at layer 30, 192 vs 158 at layer 53), so B is not handicapped by size.

Reading: detection is not attention-specific. A same-size addition of paragraph content into the residual stream raises "Yes" exactly as much as the attention boost. Localisation is attention-specific: the attention boost makes Qwen pick the boosted paragraph far more often than the content addition at the same size, or at twice the size.

Figure: figures/salience.py → figures/salience.png / .jpg (only the honest arms).
