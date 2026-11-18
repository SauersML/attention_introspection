# Attention introspection and attention control in Qwen3 (8B; 32B below)

Code: `attn_intro.py` (runs everything on a pod), `analyze.py` (tables), `run.sh`. Raw per-item records and tables:
`qwen3-8b/` (A.json, A2.json, B.json, calibration.json, results.json, tables.md) and `qwen3-32b/` (same files).
Pods: hypha-prod-attn8b (A40), hypha-prod-attn32b (A100-80GB); smoke tests hypha-prod-attnsmoke, -attnsmoke2.

## Answer in brief
- (A) No privileged access in either model: the self-report never beats the best transcript-only baseline (a classifier on transcript features, or always naming the most common position), and adding it to that classifier never helps. It beats the same model reading the transcript only once (32B, arithmetic), and that is a recency prior, not item-level information.
- (A2) A direct causal boost of attention on one paragraph (2.3–5.3× mass, making it the most-attended paragraph in up to 100% of items, with the answer text unchanged in 99.5–100% of arithmetic items) never moves the report toward that paragraph; the trace that does reach the report lowers, not raises, its probability.
- (B) Instructed control is real but small, and grows with scale: "focus on Pk" raises attention on Pk 10–40% (8B) and 10–26% (32B) over a matched mention of Pk; "do not attend" does nothing beyond naming Pk at 8B and lowers it 5–10% at 32B in arithmetic. No polar-bear effect relative to no instruction; relative to a bare mention, 32B QA shows a 7% rebound at the first answer query.

## Question
(A) Can the model say which part of its context it was attending to, better than a reader who only sees the transcript?
(A2) If we change its attention directly, does its report follow the change?
(B) If told to attend to / not attend to a paragraph, does its actual attention change?

## Prior work and what is new here
- Anthropic, "Emergent introspective awareness in LLMs" (Lindsey 2025): concept injection into the residual stream, and "think about X / don't think about X" control of a concept's representation. Ours: the object is the attention pattern itself (native softmax weights), not a concept direction in the residual stream.
- Declarative Attention (Ho et al., arXiv 2609.02737): the model declares a region in its chain of thought and the inference engine masks the KV cache. That is attention set by an external mask, not the model's native attention; it does not test whether native attention follows the declaration. Our (B) measures native attention under an instruction, with no masking.
- HKU, "A Mechanistic Study of Language Model Introspection" (arXiv 2609.35108): gate and router heads localise an injected residual vector. Ours intervenes on attention logits, not on residual activations, and asks about attention.
- Singh / Linzen / Ravfogel (arXiv 2605.26242): input-only classifiers match self-prediction, so a self-report must beat a transcript-only baseline. We use two: the same model reading the transcript, and a cross-validated classifier on transcript features (position, word overlap of task and answer with each paragraph, length).
- Also related: Ferrara, OWMI (arXiv 2608.20569): no model detects interventions on residual sites / heads / SAE features above chance against sham and text-only-observer nulls; "Detecting the Disturbance" (arXiv 2512.12411): localisation of injected sentences works only for early-layer injections; PASTA / AutoPASTA (Zhang et al., arXiv 2409.10790): adding to attention scores on a highlighted span, which is the intervention we use in A2.
- New here: native attention weights as the ground truth for an introspective report, a direct causal intervention on attention logits (with text held identical so a transcript reader is blind to it by construction), and a "mention" control that separates the effect of an instruction's meaning from the effect of merely naming the paragraph.

## Setup (identical for both models)
- Contexts: 4 SQuAD-train paragraphs (60–110 tokens each, from 4 different Wikipedia articles), labelled [P1]..[P4], then a task. Chat template, thinking off, greedy decoding.
- Three tasks, 200 contexts each: QA (a SQuAD question about one paragraph Pj), summarize all four in one sentence, and an unrelated arithmetic question ("what is 37 + 52?") for which the paragraphs are irrelevant.
- Attention measure: attention weights recomputed in a patched eager attention function, averaged over heads, summed over a paragraph's body tokens (label excluded) and divided by its token count. "First query" = the query at the last prompt token (the one that predicts the first answer token); "mean" = mean over the answer-token queries. Band = layers 12–30 of 36 for 8B; for 32B the same depth fraction, layers 21–53 of 64. Per-layer numbers are in results.json and tables.md.
- Report: after the answer, a new user turn "Which paragraph were you paying the most attention to just now, while writing your answer? Answer with just P1, P2, P3, or P4."; the assistant turn is prefilled with "P" and we read the probabilities of 1–4 (no sampling). The transcript reader is the same model given the whole user message and the answer as quoted text and asked the same question about "the assistant".
- A2 intervention: add +b to the pre-softmax attention scores of every key in paragraph Pk (k random, never the QA target paragraph), all heads, band layers, for every query from the start of the task text through the end of the answer (generation and the report pass), and not for the report question's queries. Why off during the report: the question is whether the model can report attention that happened during the task. With the bias on during the report, the report could reflect its attention at report time instead of the past. With it off, the only channel from the intervention to the report is what the task and answer tokens wrote into the KV cache, plus the answer text, which we control for. b was calibrated on 30 contexts (teacher-forced): b=1 gives about 2.3× mass on Pk (the "doubling" dose), b=2 gives about 5×.
- B: an instruction line before the task: "While you answer, focus your attention on [Pk]." / "While you answer, do not pay attention to [Pk]." / a control that names Pk with no request: "Note that one of the paragraphs is labelled [Pk]." / none. k ≠ the question's paragraph. Paired by context against "none"; ratios are geometric means of per-context mass ratios with 95% bootstrap CIs (10,000 resamples over contexts).

## Results: Qwen3-8B

### A. Report vs transcript reader (truth = most-attended paragraph at the first answer query, layers 12–30; n=200 per task)

| task | self-report | transcript reader (same model) | transcript-feature classifier (CV) | task-relevant paragraph | majority position | self − reader (paired) |
|---|---|---|---|---|---|---|
| QA | 95.5% [92.5, 98.0] | 95.5% [92.5, 98.0] | 95.0% [91.5, 98.0] | 95.5% [92.5, 98.0] | 27.5% | 0.0% [0.0, 0.0] |
| summarize | 50.0% [43.0, 57.0] | 78.0% [72.0, 83.5] | 96.5% [93.5, 99.0] | – | 96.5% (P1) | −28.0% [−34.5, −21.5] |
| arithmetic | 49.0% [42.0, 56.0] | 51.5% [44.5, 58.5] | 55.0% [48.0, 62.0] | – | 49.0% (P4) | −2.5% [−7.0, 2.0] |
| all 600 | 64.8% [61.0, 68.7] | 75.0% [71.5, 78.5] | 75.5% [72.2, 79.0] | – | 54.7% | −10.2% [−13.0, −7.5] |

Chance is 25%. With truth taken as the mean over answer queries the picture is the same (all 600: self 64.0% [60.2, 67.8], reader 67.5% [63.7, 71.3], self − reader −3.5% [−6.0, −1.0]). Adding the self-report to the transcript classifier does not raise its accuracy (all 600: 80.3% → 81.3%, CIs overlap). Taking the most-attended paragraph at any single layer instead of the band, self − reader is never above +0.02 at any of the 36 layers in any task (all 600: between −0.11 and +0.01).

**A, one sentence:** Qwen3-8B's report of where it attended carries no information beyond its transcript: it ties the transcript reader where attention is obvious from the text (QA) and is worse than it elsewhere (summarize: it says P1 half the time when the first-query attention is on P1 97% of the time).

### A2. Causal: boost attention on a non-target paragraph Pk (n=200 per task and dose)

| | QA b=1 | QA b=2 | arith b=1 | arith b=2 |
|---|---|---|---|---|
| attention mass on Pk, boost / no boost (first query) | 2.30× [2.28, 2.32] | 4.88× [4.79, 4.97] | 2.33× [2.31, 2.36] | 5.07× [4.97, 5.16] |
| Pk becomes the most-attended paragraph | 1.5% → 7.5% | 1.5% → 39.0% | 29.0% → 84.0% | 29.0% → 94.0% |
| self-report picks Pk, no boost → boost | 0.0% → 0.0% | 0.0% → 0.0% | 26.0% → 26.5% | 26.0% → 26.0% |
| transcript reader picks Pk | 0.0% → 0.0% | 0.0% → 0.0% | 27.5% → 27.5% | 27.5% → 27.5% |
| difference-in-differences (self − reader) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 0.5% [−2.5, 3.5] | 0.0% [−3.5, 3.5] |
| Pk-specific shift in the report's log-probability, self | −0.26 [−0.31, −0.21] | −0.60 [−0.70, −0.50] | −0.28 [−0.40, −0.17] | −0.44 [−0.71, −0.17] |
| same, transcript reader | −0.005 [−0.024, 0.013] | −0.003 [−0.031, 0.027] | 0.000 | 0.000 |
| answer text identical to no-boost | 87.5% | 73.5% | 100% | 100% |
| same shift, identical-answer subset only | −0.26 [−0.31, −0.21] | −0.57 [−0.68, −0.47] | −0.28 [−0.40, −0.17] | −0.44 [−0.71, −0.17] |
| text leak: change in answer's word overlap with Pk | −0.001 [−0.002, 0.000] | −0.001 [−0.005, 0.001] | 0.000 | 0.000 |
| QA / arithmetic accuracy, no boost → boost | 81.5% → 81.0% | 81.5% → 82.0% | 100% → 100% | 100% → 100% |

"Pk-specific shift" = change in log P(report = Pk) minus the mean change in log P(report = m) over the other non-target paragraphs m; 0 is the null for a perturbation that does not single out Pk. On the identical-answer subset the transcript reader sees byte-identical text, so its shift is exactly 0 and there is no text leak.

**A2, one sentence:** Doubling or quintupling attention on a paragraph (enough to make it the most-attended paragraph in 84–94% of arithmetic items) never moves the self-report toward it; the intervention does reach the report through the KV cache, with text held identical, but it lowers the probability of naming the boosted paragraph rather than raising it.

### B. Instructed attention to Pk (k ≠ question paragraph; n=200 contexts per task, paired with no instruction)

| task | condition | mass on Pk vs none, first query | mass on Pk vs none, mean over answer | mass on question paragraph Pj (mean) | answer identical to none | correct |
|---|---|---|---|---|---|---|
| QA | focus | 0.99× [0.97, 1.01] | 1.08× [1.05, 1.12] | 0.88× [0.83, 0.93] | 55.0% | 77.5% |
| QA | ignore | 0.85× [0.84, 0.86] | 0.98× [0.97, 0.99] | 1.02× [1.00, 1.05] | 65.0% | 82.5% |
| QA | mention only | 0.86× [0.85, 0.87] | 0.99× [0.97, 1.00] | 0.99× [0.97, 1.00] | 67.0% | 82.5% |
| arith | focus | 1.17× [1.13, 1.21] | 1.03× [1.01, 1.05] | 0.89× [0.87, 0.91] | 97.5% | 97.5% |
| arith | ignore | 0.83× [0.82, 0.85] | 0.91× [0.89, 0.92] | 0.92× [0.91, 0.93] | 100% | 100% |
| arith | mention only | 0.84× [0.82, 0.86] | 0.93× [0.91, 0.94] | 0.86× [0.85, 0.88] | 99.0% | 99.0% |

(none: QA correct 81.5%, arith 100%.) Paired contrasts on mass on Pk:
- focus vs mention: QA 1.14× [1.12, 1.16] first query, 1.10× [1.07, 1.13] mean; arith 1.39× [1.36, 1.42] first, 1.11× [1.11, 1.12] mean.
- ignore vs mention: QA 0.987× [0.980, 0.994] first, 0.993× [0.984, 1.004] mean; arith 0.992× [0.985, 0.998] first, 0.980× [0.973, 0.987] mean.
- Per layer (mean over answer queries): in QA the focus effect is positive from layer 13 to 34 (e.g. +0.17 log-ratio at layer 18, +0.16 at 24); in arithmetic it is positive at layers 13–18 (+0.16 at 13, 15 and 17) and mixed after. Ignore and mention track each other within 0.1 at every layer (mostly within 0.05).

**B, one sentence:** Qwen3-8B has weak but real instructed control over its native attention: "focus on Pk" raises attention on Pk by about 10–40% over a matched mention of Pk (most at the first answer query in the arithmetic task, where the answer text is unchanged in 97.5% of items), while "do not attend to Pk" does almost nothing beyond merely naming Pk (≤2% further decrease); the decrease seen versus no instruction is a naming effect, and there is no polar-bear rebound (ignore is not above none).

## Results: Qwen3-32B (layers 21–53 of 64; same contexts, same code; b=1 gives 2.4×, b=2 gives 5.2× mass)

### A. Report vs transcript reader (n=200 per task)

| task, truth | self-report | transcript reader | transcript-feature classifier (CV) | classifier + reader + self (CV) | majority position | self − reader (paired) |
|---|---|---|---|---|---|---|
| QA, first query | 92.5% [88.5, 96.0] | 92.5% [89.0, 96.0] | 92.5% [88.5, 96.0] | 92.5% | 29.5% | 0.0% [0.0, 0.0] |
| summarize, first query | 48.0% [41.0, 55.0] | 67.0% [60.5, 73.5] | 93.0% [89.0, 96.5] | 93.0% | 93.0% (P1) | −19.0% [−25.5, −12.5] |
| arithmetic, first query | 34.0% [27.5, 40.5] | 30.5% [24.0, 37.0] | 39.5% [33.0, 46.0] | 39.5% | 38.0% (P4) | +3.5% [−4.5, 11.5] |
| all 600, first query | 58.2% [54.2, 62.0] | 63.3% [59.3, 67.2] | 71.0% [67.5, 74.7] | 73.8% | 49.8% | −5.2% [−8.7, −1.7] |
| arithmetic, mean over answer | 54.0% [47.0, 61.0] | 26.0% [20.0, 32.0] | 69.5% [63.0, 75.5] | 69.5% | 69.5% (P4) | +28.0% [20.5, 36.0] |
| all 600, mean over answer | 57.3% [53.3, 61.2] | 48.0% [44.0, 52.2] | 75.8% [72.3, 79.2] | 76.0% | 51.8% | +9.3% [6.2, 12.5] |

The one place 32B's self-report beats the reader (arithmetic, truth over all answer queries) is a positional prior, not item-level access: the self-report says P4 in 141/200 items, attention's argmax is P4 in 139/200, and the reader spreads its answers (55/200 P4). The self-report is below simply always saying P4 (54.0% vs 69.5%), and adding it to the transcript classifier adds nothing (69.5% → 69.5%).

**A (32B), one sentence:** Qwen3-32B's report also carries no item-level information about its attention beyond the transcript: where it beats the transcript reader it does so by a recency prior that "always P4" or a transcript classifier beats outright, and adding the report to the classifier never helps (all 600: 75.8% → 76.0%).

### A2. Causal boost (n=200 per task and dose)

| | QA b=1 | QA b=2 | arith b=1 | arith b=2 |
|---|---|---|---|---|
| attention mass on Pk, boost / no boost (first query) | 2.40× [2.39, 2.42] | 5.27× [5.21, 5.33] | 2.38× [2.37, 2.40] | 5.22× [5.16, 5.29] |
| Pk becomes the most-attended paragraph | 1.5% → 38.5% | 1.5% → 99.5% | 25.5% → 88.5% | 25.5% → 100% |
| self-report picks Pk, no boost → boost | 0.0% → 0.0% | 0.0% → 0.0% | 25.5% → 25.0% | 25.5% → 23.0% |
| transcript reader picks Pk | 0.0% → 0.0% | 0.0% → 0.0% | 18.5% → 18.5% | 18.5% → 18.5% |
| difference-in-differences (self − reader) | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | −0.5% [−3.5, 2.5] | −2.5% [−6.0, 1.0] |
| Pk-specific log-prob shift, self | −0.21 [−0.25, −0.18] | −0.45 [−0.53, −0.38] | −0.27 [−0.32, −0.21] | −0.36 [−0.49, −0.24] |
| same, transcript reader | −0.002 [−0.021, 0.014] | −0.009 [−0.048, 0.026] | 0.000 | −0.003 [−0.008, 0.000] |
| answer text identical to no-boost | 87.5% | 71.5% | 100% | 99.5% |
| same shift, identical-answer subset | −0.22 [−0.26, −0.18] | −0.46 [−0.55, −0.37] | −0.27 [−0.32, −0.21] | −0.36 [−0.49, −0.24] |
| text leak: change in answer word overlap with Pk | −0.000 [−0.001, 0.000] | −0.001 [−0.003, 0.001] | 0.000 | 0.000 |
| accuracy, no boost → boost | 83.0% → 82.5% | 83.0% → 83.0% | 99.5% → 99.5% | 99.5% → 100% |

**A2 (32B), one sentence:** Same as 8B: making Pk the most-attended paragraph in up to 100% of items never moves the report toward Pk, and the trace that does reach the report through the KV cache (with the text identical) again lowers the probability of naming Pk.

### B. Instructed attention (n=200 contexts per task)

| task | condition | mass on Pk vs none, first query | mass on Pk vs none, mean over answer | mass on Pj (mean) | answer identical to none | correct |
|---|---|---|---|---|---|---|
| QA | focus | 0.93× [0.92, 0.93] | 1.11× [1.09, 1.12] | 0.92× [0.90, 0.95] | 49.0% | 84.5% |
| QA | ignore | 0.90× [0.89, 0.90] | 0.98× [0.97, 0.99] | 0.99× [0.98, 1.00] | 62.0% | 85.0% |
| QA | mention only | 0.84× [0.83, 0.84] | 1.00× [0.99, 1.01] | 0.97× [0.97, 0.98] | 63.5% | 84.5% |
| arith | focus | 1.03× [1.02, 1.05] | 1.39× [1.35, 1.43] | 0.99× [0.97, 1.01] | 99.0% | 99.0% |
| arith | ignore | 0.88× [0.88, 0.89] | 0.99× [0.98, 1.01] | 0.96× [0.95, 0.97] | 100% | 99.5% |
| arith | mention only | 0.93× [0.92, 0.94] | 1.10× [1.08, 1.12] | 0.96× [0.95, 0.98] | 100% | 99.5% |

(none: QA correct 83.0%, arith 99.5%.) Paired contrasts on mass on Pk:
- focus vs mention: QA 1.11× [1.10, 1.12] first, 1.10× [1.09, 1.12] mean; arith 1.11× [1.10, 1.12] first, 1.26× [1.25, 1.28] mean.
- ignore vs mention: QA 1.07× [1.07, 1.08] first (ignore keeps more attention on Pk than a bare mention), 0.975× [0.969, 0.981] mean; arith 0.95× [0.94, 0.96] first, 0.90× [0.90, 0.91] mean.
- Per layer (arith, mean over answer, vs no instruction): focus is positive at every layer from 21 to 63, largest at layers 24, 26 and 28 (+0.75, +0.78, +0.95 log-ratio, i.e. 2.1–2.6×); mention reaches +0.34 at 24 and +0.35 at 52; ignore stays within −0.16 to +0.17.

**B (32B), one sentence:** At 32B instructed control is stronger and partly two-sided: "focus on Pk" raises attention on Pk by 10–26% over a matched mention (up to 2.6× over no instruction in single layers 24–28, answer text unchanged in 99% of arithmetic items), and "do not attend" now lowers it 5–10% below a mention in the arithmetic task; but in QA at the first answer query "do not attend" leaves 7% more attention on Pk than a bare mention, a small polar-bear effect relative to the mention control (both are still below no instruction).

## Scale comparison (8B → 32B)
- Introspective report (A, A2): no change. Neither model's report tracks its attention beyond the transcript, and in both the attention boost moves the report's log-probability for the boosted paragraph down by a similar amount (−0.2 to −0.6).
- Control (B): grows with scale. Focus vs mention on mean answer-query attention, arithmetic: 1.11× (8B) → 1.26× (32B). Ignore vs mention, arithmetic, mean: 0.98× → 0.90×.

## Cost
Pods: $3.49 total from the ledger (8B on A40 $0.37, 32B on A100-80GB $2.93, two smoke tests and discarded offers $0.19).

## Caveats
- One prompt wording for the report question, the reader question and each instruction; one paragraph format; greedy decoding. Different wording could change levels, though the A2 test (text identical, report moves the wrong way) does not depend on the reader's wording.
- "Most-attended paragraph" is a head-averaged, layer-band-averaged, length-normalised summary. The model has no reason to know this particular aggregate; a report could track some other aggregate (specific heads, last layers, value-weighted attention). For 8B, self − reader is never above +0.02 at any single layer (first query); for 32B it reaches +0.43 at some layers in arithmetic, most likely the same P4 prior that the classifier test exposes in the band (not checked layer by layer). A head subset could still hold a match.
- QA leaves no room for privileged access: argmax attention equals the question's paragraph 92.5–100% of the time, which the text gives away. Summarize and arithmetic are where the report could have beaten the transcript baselines; it never beat the best of them.
- A2's intervention is outside the model's normal operating range (a uniform logit bias across all heads in 19 layers for 8B, 33 for 32B). A report that ignores an unnatural change does not prove the model cannot read natural variation; A (natural variation) and A2 agree, though.
- The negative Pk-specific shift in A2 is unexplained. It is in log-probability only (the rate of naming Pk does not rise; it stays within 2.5 points), and its sign is the opposite of introspection. One untested guess: the extra Pk content written into the task and answer positions makes the report question's queries treat Pk as already-covered context. Untested: a sham bias of the same size on non-paragraph tokens.
- In A2 the bias also applies to the task-question tokens' queries, so the task tokens' KV carries Pk content too; that is part of "attention during the task".
- In B, instructions change the answer text in QA (35–51% of answers differ in wording from no instruction; correctness stays within 4 points of no instruction, CIs overlapping), so QA attention differences mix control with content; the arithmetic task, where answers are unchanged in 97.5–100% of items, is the clean read.
- Bootstrap CIs are over contexts; the paragraphs are SQuAD Wikipedia paragraphs, so "neutral content" means encyclopedic text.
