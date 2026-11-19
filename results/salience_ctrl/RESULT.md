# Salience controls (Qwen3-32B, 240 contexts, paired per context)

**Misleading prompt, not used:** this prompt tells Qwen that only attention boosts happen, so the content-addition (B, B2) and random-direction (R) arms here misled Qwen; their data stay in data/ but are not used in any figure or claim. Only the honest arms (none, A, and the pick-at-random question QA) are used, in figures/salience.py.

Same told prompt and arms as results/salience ("I am capable of artificially boosting your attention onto one of the four paragraphs below … Do you notice your attention being boosted onto one of the paragraphs? If so, which one?"), layers 21–53, at the question and answer tokens, each paragraph k boosted in turn (boosted arms averaged over k per context):
- none; A: attention boost ×20; B: paragraph k's content direction added into the residual stream, same size as A's residual change; B2: 2× B; R: a random direction per layer, same size as B.
- Qnone / QA: no boost / the same attention boost ×20, but the question "Pick one of the four paragraphs at random" (prefill "paragraph [P").

Means over contexts, 95% bootstrap CIs over contexts (10,000 resamples), p two-sided sign-flip over contexts on paired differences (20,000 flips; 5e-05 is the floor). Computed by figures/salience_controls.py (removed; A and QA now in figures/pick_random.py).

Greedy reply starts "Yes":
- none 12.9% [8.8, 17.5]; A 22.1% [19.0, 25.2]; B 21.4% [18.9, 23.9]; B2 17.4% [15.2, 19.6]; R 16.4% [13.3, 19.5]
- A vs B p = 0.72; A vs R p = 0.001; A vs none p = 5e-05; B vs none p = 2e-04; R vs none p = 0.053

Top digit after the prefill "Yes, paragraph [P" is the boosted paragraph (chance 25%):
- A 58.3% [55.5, 61.1]; B 35.1% [33.0, 37.2]; B2 38.9% [36.4, 41.4]; R 25.0% [24.2, 25.8]
- A vs B p = 5e-05; B vs R p = 5e-05

Same attention boost, asked to pick a paragraph at random (top digit after "paragraph [P" is the boosted one): QA 37.2% [35.2, 39.2] vs 58.3% when asked about the boost; p = 5e-05.

Reading: a random direction of the same size raises "Yes" less than the attention boost or the content addition (16% vs 22%/21%) and leaves the paragraph choice at chance, so the localisation is not a generic perturbation effect. The attention boost also biases a "pick at random" answer toward the boosted paragraph (37%), but much less than when Qwen is asked about the boost (58%), so most of the localisation depends on the question asking about the boost.

Figure: figures/pick_random.py → figures/pick_random.png / .jpg (A vs QA only).
