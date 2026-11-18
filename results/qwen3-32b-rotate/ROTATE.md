# Rotation control for instructed attention (Qwen3-32B, arithmetic task), and a sham null for A2

Code: `rotate.py` (both experiments, runs on a pod), `rotate_analyze.py`, `sham_analyze.py`, `rotate_run.sh`.
Data: `qwen3-32b-rotate/` (R.json = per-context rotation records, SHAM.json = sham records, rotate_results.json, sham_results.json, run.log).
Figure: `compute/fig/attn_intro/rotate.png` / `.jpg` (script `compute/fig/attn_intro/rotate_fig.py`).
Pod: hypha-prod-attnrot32b (A100-SXM4-80GB, 0.50 h, $1.10); a first launch (hypha-prod-attnrot32, H100, $1.26) failed before running because its scripts were not uploaded. Total about $2.5 including discarded pod creations. All pods deleted (confirmed in the runner log).

## Question
Experiment B showed "focus on Pk" raises attention on Pk. Is that specific to the paragraph named, or does the instruction raise attention on paragraphs generally (or on a fixed favourite)? Rotate the named paragraph over all four on the same context.

## Setup
- 100 contexts built exactly as in attn_intro.py (4 SQuAD-train paragraphs of 60–110 tokens from 4 different articles, labelled [P1]..[P4], then "Unrelated question: what is X + Y? Answer with just the number."), seed 2 (the original used seed 0).
- 5 versions per context, differing only in one line placed before the question: none, or "Focus your attention on paragraph [Pk]." for k = 1..4. Same paragraphs, order and question in all 5.
- Measure as in B: attention weights from a patched eager attention function, averaged over heads, summed over each paragraph's body tokens (label excluded) and divided by its token count, averaged over the answer-token queries and over layers 21–53 of 64.
- Teacher forcing: all 5 versions are scored on the no-instruction answer's tokens. Free (greedy) answers under the instruction were also generated: 398 of 400 are identical to the no-instruction answer (the 2 that differ are one context, "134" vs "94", under the P1 and P4 instructions). No-instruction accuracy 99%.
- Statistic per cell: geometric mean over contexts of (attention on measured paragraph with instruction k) / (same, no instruction); 95% bootstrap CI over contexts (10,000 resamples).

## Result: mean over answer queries, layers 21–53 (the B measure)

Ratio to no instruction (rows: paragraph named in the instruction; columns: paragraph measured):

| named \ measured | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | **1.826 [1.772, 1.884]** | 1.172 [1.156, 1.187] | 0.980 [0.966, 0.996] | 0.781 [0.769, 0.793] |
| P2 | 1.261 [1.235, 1.288] | **1.810 [1.754, 1.868]** | 1.244 [1.222, 1.268] | 0.867 [0.852, 0.881] |
| P3 | 1.000 [0.981, 1.020] | 1.377 [1.346, 1.410] | **1.704 [1.662, 1.747]** | 1.067 [1.047, 1.087] |
| P4 | 0.808 [0.794, 0.823] | 1.009 [0.996, 1.022] | 1.116 [1.101, 1.134] | **1.273 [1.254, 1.293]** |

- The diagonal is the largest cell in every row. Diagonal ÷ largest off-diagonal in the same row (paired bootstrap): P1 1.56 [1.51, 1.61], P2 1.44 [1.39, 1.48], P3 1.24 [1.20, 1.27], P4 1.14 [1.12, 1.16]. All four CIs exclude 1.
- Per context: the fraction of the 4 instructions for which the named paragraph got the largest boost (log ratio) of the four: mean 0.97 [0.95, 0.985]. 88 contexts at 4/4, 12 at 3/4, none below. By instruction: P1 100%, P2 100%, P3 92%, P4 96%.
- The boost spills onto neighbouring paragraphs (P2 named: P1 1.26×, P3 1.24×) and is weakest for P4, the paragraph next to the question. Naming P1 lowers P4 to 0.78× and naming P4 lowers P1 to 0.81×.

Same matrix in SD units ((attention with instruction − attention without) ÷ SD across contexts of no-instruction attention on the measured paragraph; mean over contexts, 95% bootstrap CI):

| named \ measured | P1 | P2 | P3 | P4 |
|---|---|---|---|---|
| P1 | **+2.88 [+2.57, +3.25]** | +0.30 [+0.27, +0.34] | −0.02 [−0.05, +0.02] | −0.42 [−0.45, −0.38] |
| P2 | +0.91 [+0.80, +1.02] | **+1.47 [+1.31, +1.65]** | +0.48 [+0.40, +0.56] | −0.24 [−0.27, −0.20] |
| P3 | +0.04 [−0.04, +0.14] | +0.68 [+0.60, +0.77] | **+1.32 [+1.19, +1.47]** | +0.16 [+0.11, +0.22] |
| P4 | −0.63 [−0.69, −0.57] | +0.04 [+0.01, +0.06] | +0.23 [+0.19, +0.28] | **+0.55 [+0.50, +0.60]** |

Diagonal largest in every row in SD units too. Per-context fraction in SD units: 0.855 (52 contexts 4/4, 38 at 3/4, 10 at 2/4). (SD units weight paragraphs by their spread across contexts, so P1, which has low and steady no-instruction attention, gets the largest values.)

## Secondary: the first answer query only (last prompt token), layers 21–53
At this single query the instruction lowers attention on all non-named paragraphs (0.82–0.91×) and leaves the named one near 1 (0.92, 1.00, 1.02, 1.00 for P1..P4). The diagonal is still the largest ratio in every row, but the boost is relative only; per-context fraction 0.86 [0.82, 0.90] (P1 instruction only 62%). The B-measure result above is the main one.

## Reading
Instructed attention in Qwen3-32B is specific to the paragraph named: rotating the name rotates the boost, in 97% of (context, instruction) pairs the named paragraph gains the most of the four, with spillover onto its neighbours.

What would overturn "the model follows the instruction's meaning": the boost could come from the instruction's "[Pk]" token matching the "[Pk]" label token, pulling attention to the label's neighbourhood by surface match, with no role for "focus". This experiment cannot separate those, because every row contains both the name and the request. Experiment B's mention control (same paragraph named, no request) addresses it and found focus > mention by 1.26× [1.25, 1.28] in 32B arithmetic, so part, not all, of the boost is the request. Null: with no effect of the named paragraph the per-context fraction is 0.25 by chance; observed 0.97. Not run here: a rotation of the mention-only line.

## Sham null for A2

**Correction (18 Nov, after review).** The sham below cannot test Pk-specificity: the header/question tokens it boosts are the same whichever paragraph is the target k, and k is random, so its expected Pk-specific shift is zero by construction; its −0.04 says nothing about specificity. Its one use is the overall-disturbance row (a non-specific boost disturbs the report as much). The real control is the label-token sham below ("Label-token sham").

### Question
In A2, boosting attention on a paragraph Pk (b=2 on the pre-softmax scores of Pk's keys) lowered the report's probability of naming Pk. Is that specific to Pk, or does any attention boost disturb the report this way?

### Setup
- Same 100 arithmetic contexts as above (no instruction line), b=2, boost applied to queries from the start of the task text through the end of the answer (generation and the report pass), not to the report question's queries, all heads, layers 21–53. Report read as in RESULT.md (prefill "P", probabilities of 1–4).
- Conditions: none; boost on Pk (k random, as in A2); SHAM: the same b on non-paragraph tokens of the user message, "Here are four paragraphs." and the question line "Unrelated question: what is X + Y? Answer with just the number." (24 tokens; no paragraph tokens, checked by assertion).
- Equal token count was not possible: the whole non-paragraph user text is 24 tokens, while Pk has 94 on average. To check the sham is not simply weaker, I measured the attention mass each boost adds at the first answer query (teacher-forced on the no-boost answer): the Pk boost adds 0.033 [0.031, 0.034] of total attention, the sham at b=2 adds 0.124 [0.124, 0.125], about 4× more. The planned mass-matched sham searched b ∈ {2, 3, 4, 5, 6, 8, 10} and picked b=2 for every context (b would need to be below 2 to match), so it is the same condition as SHAM (re-run; its numbers differ only through generation).
- Pk-specific shift (as in RESULT.md): change in log P(report = Pk) minus the mean change in log P(report = m) over the other three paragraphs m, versus no boost. 0 means the perturbation does not single out Pk.

### Result (n = 100)

| | boost on Pk | sham (question and header text) |
|---|---|---|
| attention on Pk, ratio to no boost (answer queries) | 4.89× | 0.45× |
| attention on sham tokens, ratio to no boost (answer queries) | 0.96× | 1.70× |
| Pk-specific shift in log P(report) | **−0.54 [−0.72, −0.36]** | **−0.04 [−0.27, +0.16]** |
| same, answers identical to no-boost | −0.54 [−0.73, −0.36] (n=99) | −0.05 [−0.28, +0.16] (n=98) |
| overall report disturbance: mean over P1–P4 of abs. change in log P | 0.51 [0.44, 0.58] | 0.57 [0.50, 0.64] |

Paired difference boost − sham in the Pk-specific shift: −0.50 [−0.78, −0.22].

### Reading
The negative shift is specific to the boosted paragraph. A sham boost on non-paragraph text that adds about 4× more attention mass, and disturbs the report's distribution at least as much overall, produces no Pk-specific shift (CI includes 0), and the difference from the Pk boost excludes 0. So the report does carry a trace of which paragraph was boosted, and that trace pushes the report away from the boosted paragraph, not toward it.

What would overturn this: a sham that touched paragraph-like content (e.g. the [Pk] label tokens) might also produce a Pk-specific shift; this sham touched no paragraph and no label. Null: the sham itself is the null; its shift is −0.04 [−0.27, +0.16].

## Label-token sham (18 Nov)

Code `sham_label.py` (rotate.py's setup verbatim, then this loop), `shamlabel_run.sh`, `sham_label_analyze.py`; data `SHAMLABEL.json`, `shamlabel_run.log`; numbers `sham_label_results.json`. Pod run hypha-prod-attnshamlabelb (A100, $0.52).

Same 100 contexts (seed 2), same b and query range as the sham above. Conditions: none; boost on Pk's body tokens (b=2, a rerun of the boost above); the same b=2, and b=4, on the label tokens "[Pk]" of the same paragraph only (4 tokens; no body token, asserted). This control depends on k exactly as the body boost does.

| | body boost b=2 | label boost b=2 | label boost b=4 |
|---|---|---|---|
| attention on Pk's body, ratio to no boost (answer queries, layers 21–53) | 4.89× | 0.94× | 0.79× |
| attention on Pk's label, ratio to no boost | 0.88× | 6.10× | 25.0× |
| Pk-specific shift in log P(report) | **−0.54 [−0.73, −0.37]** | **+0.13 [+0.03, +0.22]** | **+0.43 [+0.23, +0.64]** |
| answers identical to no boost | 99 of 100 | 99 of 100 | 99 of 100 |

Body − label (b=2), paired: −0.67 [−0.91, −0.42].

Reading: the rerun reproduces the body boost (−0.54). Boosting attention on the same paragraph's label moves the report the other way, toward naming Pk, and more at the larger bias. So the inverted trace is tied to attention on the paragraph's text, not to any perturbation located at paragraph k. The body boost also lowers attention on the label (0.88×); since label attention pushes the report toward Pk, part of the inverted trace may run through the label. Not tested.
