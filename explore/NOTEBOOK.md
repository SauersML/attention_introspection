# Lab notebook: attention control in Qwen3-32B (overnight, 18 Nov 2026)

Run serially: each experiment chosen from what the last one showed. Model Qwen3-32B (64 layers, 64 query heads), one A100 80 GB pod, eager attention with a hook that records per-head attention from the answer tokens to named token spans. Unless said otherwise: four SQuAD paragraphs (60–110 tokens, a fresh seed-11 draw, not the contexts of the earlier study), an unrelated sum, the model's greedy no-instruction answer teacher-forced in every condition; metric = attention per paragraph token, mean over all heads of layers 21–53, as a per-prompt log ratio against that condition's matched control; reported as percent with 95% bootstrap CI over prompts and a sign-flip p. "Specific" = target paragraph's log ratio minus the mean of the other three (so a boost that lifts every paragraph scores 0).

## Prior work (searched 04:20)
PRIOR.md (results/) covers it: Zhang, Meng & Collier 2406.17095 (document-ID attention instructions raise attention to that document; relative-position words like "middle part" fail), MFAI 2601.12499 (deep-layer concentration, Qwen2.5-7B), PASTA/SpotLight/InstABoost/SEKA/Prism-Δ (external attention steering, not instructed). A new search (Exa, "instructed attention ... measured attention weights", 2025–26) found only more external steering methods (Prism-Δ 2603.10705, SEKA ICLR 2026, Prefix Steering 2610.04967). Nothing found on: addressing by topic/quote without labels, instructions in other languages/encodings, graded ratios, mid-answer switching, or which heads carry the instruction. Those are open.

## E1 — Does the focus instruction need the [Pk] label? (started 04:25 by the previous agent, analysed 04:55)
**Why.** The mechinterp result says six heads in layers 20–26 copy the instruction's digit forward. If the boost lives in digit copying, addressing a paragraph any other way (ordinal, topic, quote) should fail or use something else; a translated or encoded label line tests whether it is the words or the label token that matters.
**Setup.** 150 contexts (seed 11), four labelled or unlabelled SQuAD paragraphs, unrelated sum, no-instruction greedy answer teacher-forced. Ten addressing modes, each a focus line and its own matched mention line that refers to the paragraph the same way but asks nothing (e.g. "Focus your attention on the paragraph about Kanye West." vs "Note that one of the paragraphs is about Kanye West."). Instruction just before the question. Script e1_address.py, analysis a1.py → out/e1/a1.json (pod).
**Control.** Matched mention per mode; "specific" = target minus mean of other three paragraphs.
**Result** (specific boost to the addressed paragraph, focus vs its mention; 95% CI; all p < 0.0001; "top" = share of prompts where the addressed paragraph gets the largest rise):

| mode | specific | top |
|---|---|---|
| label, English | +38.5% [35.5, 41.8] | 0.97 |
| label, French | +39.4% [36.5, 42.6] | 0.96 |
| label, Chinese | +38.3% [34.8, 42.1] | 0.97 |
| ordinal, labels present | +30.9% [27.5, 34.5] | 0.93 |
| ordinal, no labels | +20.3% [17.6, 23.1] | 0.79 |
| quote of first 5 words, no labels | +9.8% [8.9, 10.8] | 0.87 |
| topic (SQuAD title), no labels | +9.3% [8.2, 10.5] | 0.80 |
| label in a rhyming couplet | +5.0% [4.2, 5.8] | 0.62 |
| label line in base64 | +4.4% [3.7, 5.2] | 0.57 |
| emoji "👀👉 [Pk]" | +2.4% [2.0, 2.7] | 0.67 |

Mention vs no instruction at all, for the addressed paragraph: label mention +7.1% [4.7, 9.5]; topic mention −6.0% [−8.0, −4.0]; quote mention −7.7% [−10.1, −5.4]; ordinal mention −1.7% [−2.9, −0.5]. Teacher-forced answer log-prob barely moves (|Δ| ≤ 0.05 nats).
**Surprises.** (1) Language doesn't matter at all (French, Chinese = English), but form does: the same label inside a couplet ("let [Pk] hold your gaze the whole way through") gives 1/8 of the plain sentence's effect. So it isn't the digit token alone; the model needs to parse an attention *instruction*. (2) Addressing by content (topic, quote) works but at a quarter of the label's strength, even though the quote literally contains the paragraph's first words. (3) Merely mentioning the topic or quote *lowers* attention to that paragraph vs no line at all, while mentioning the label raises it — a copy of the paragraph's words near the answer seems to act as a decoy that soaks up the attention the paragraph would get.
**Next.** (a) Head profiles: is the same downstream machinery used for every addressing mode (a1b.py, no new forwards). (b) The previous studies all used an unrelated sum, where attention to the paragraphs is useless. Does instructed attention help or hurt when the question is actually about one paragraph? → E2.

### E1b — Same heads for every way of addressing? (a1b.py, no new forwards; out/e1/a1b.json)
Per-head specific effect (layers 21–53, 2112 heads, mean over 150 prompts). Split-half reliability of each mode's head profile is 0.87–0.99, so correlations near that are "same heads".
- Label English/French/Chinese: r = 0.98 with each other (at ceiling). Ordinal with labels 0.95, ordinal without labels 0.86, topic 0.79, quote 0.78, emoji 0.73, couplet 0.64. **base64: 0.07**, though its own profile is reliable (split-half 0.92): the small base64 effect runs through different heads, so it is not a weak copy of the instruction.
- Topic and quote correlate 0.95 with each other and 0.92 with unlabelled ordinal: content addressing shares the machinery, just drives it ~4× less.
- **The six "digit" heads (20.19, 20.29, 23.61, 25.28, 25.30, 26.5) are not digit heads.** Their attention from the answer tokens to the instruction line, focus vs mention: label 0.41 vs 0.19; French 0.42 vs 0.19; Chinese 0.46 vs 0.23; ordinal without labels 0.31 vs 0.05; topic 0.22 vs 0.11; quote 0.28 vs 0.13; couplet 0.22 vs 0.18; emoji 0.27 vs 0.25; base64 0.05 vs 0.08. They read the address of an attention instruction whatever its form, and the modes where they don't distinguish focus from mention (couplet, emoji, base64) are the ones that barely work. (Mean over prompts, no CIs yet; ordinal is the outlier: strong reading, middling effect.)
