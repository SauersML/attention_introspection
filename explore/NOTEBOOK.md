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

## E2 — Does instructed attention help or hurt when the question needs one paragraph? (05:00–05:30)
**Why.** Every result so far used an unrelated sum, where attention to the paragraphs does nothing for the answer. The obvious question is what the instruction does when the task itself directs attention: does focusing on the right paragraph help, does focusing on a wrong one hurt?
**Setup.** 150 contexts (seed 11), four labelled paragraphs; the question is the SQuAD question for paragraph k ("…Answer with just the answer, a few words."). Seven conditions: none; focus / mention / ignore naming the right paragraph [Pk]; the same naming a wrong one [Pk2]. Gold answer teacher-forced for attention and log-prob; greedy answer generated per condition for accuracy (gold string contained, normalised). e2_help.py, a2.py → out/e2/a2.json.
**Control.** No instruction, plus the matched mention lines (any extra line raises the gold log-prob by ~0.4 nats by itself, so log-prob comparisons are against mention).
**Result.**
- The question already concentrates attention: per-token attention (layers 21–53) on the answer paragraph is 6.7× that on each other paragraph (11.2 vs 1.67 ×10⁻⁴), and up to 20–30× in layers 45–52.
- Focus on the right paragraph adds nothing to it: +1.0% [0.4, 1.6]; it mainly lowers the others (−8.0% [−9.3, −6.7] on the named wrong one).
- Focus on a wrong paragraph raises it +17.2% [15.1, 19.3] (vs +47% when the task is an unrelated sum) and takes only −4.6% [−5.3, −3.8] from the answer paragraph. In absolute terms it adds 0.27×10⁻⁴ per token, 3% of what the question puts on the right paragraph. Its boost sits in layers 21–28 (peaks +52–56% at 26 and 28), weaker after.
- Accuracy: none 0.73 [0.65, 0.79]; focus right 0.75; focus wrong 0.71 [0.64, 0.78]; no condition differs. Gold log-prob, focus minus mention: right −0.23 [−0.43, −0.03] (p = 0.03), wrong +0.20 [−0.06, +0.48]. Per prompt, the attention change on the answer paragraph does not predict the log-prob change (r = −0.01).
**Surprises.** The instruction is a small additive nudge on top of task-driven attention, not a controller: it can't raise attention that the task already maxed, and pointing it elsewhere doesn't pull attention away from where the answer is. "Focus on the right paragraph" even slightly lowers the gold answer's log-prob (weak, p = 0.03).
**Next.** A case where the instruction could matter: two paragraphs give *conflicting* answers to the same question. Does "focus on [Pj]" choose which source wins, and how does that compare with an explicit "answer using only [Pj]"? → E4 (after E3, already queued: which verbs drive the six heads, and is ablating them causal for unlabelled addressing).

## E3 — Which instructions drive the six heads, and are they causal beyond digit labels? (05:15–05:40)
**Why.** E1b: the six heads read the address of attention instructions in any form (ordinal, topic, French). Is that an "attend" reader, or a generic pointer to whatever paragraph the instruction is about? And is it causal when there is no label digit to copy?
**Setup.** 150 contexts (seed 11), labelled paragraphs, unrelated sum, no-instruction answer teacher-forced. Eleven lines naming [Pk] with different verbs, all against the same mention line ("Note that one of the paragraphs is labelled [Pk]."). Ablation: zero the six heads' outputs (all positions) vs a random six from layers 20–26 (fresh per prompt), under label, unlabelled-ordinal and unlabelled-topic focus/mention. e3_verbs.py, a3.py → out/e3/a3.json.
**Control.** Matched mention; random-six ablation.
**Result** (specific boost to [Pk]; six-head attention to the instruction line, minus mention):

| line | specific | six-head reading |
|---|---|---|
| Focus your attention on paragraph [Pk]. | +38.5% [35.5, 41.8] | +0.22 |
| After answering, summarize paragraph [Pk]. | +36.3% [31.8, 41.3] | +0.32 |
| Read paragraph [Pk] carefully. | +32.7% [29.4, 36.2] | +0.22 |
| Pay attention to the label [Pk]. | +17.2% [15.7, 18.8] | +0.14 |
| Paragraph [Pk] is the important one. | +8.6% [7.2, 10.0] | +0.05 |
| Paragraph [Pk] contains a typo. | +8.5% [7.3, 9.9] | +0.05 |
| Paragraph [Pk] was written by my friend. | +7.3% [5.8, 8.9] | +0.06 |
| Memorize paragraph [Pk]; I will ask about it later. | +6.5% [4.9, 8.2] | +0.07 |
| Do not think about paragraph [Pk]. | +4.9% [3.7, 6.1] | +0.02 |
| Completely ignore paragraph [Pk]. | −6.2% [−7.2, −5.3] | −0.06 |
| Skip paragraph [Pk]; it is irrelevant. | −6.5% [−7.3, −5.8] | −0.04 |

Per prompt, pooled over the eleven lines, the six heads' extra reading of the instruction predicts the specific boost: r = 0.76.
Ablation (specific boost, focus vs mention): label +38.5% → +14.0% [12.8, 15.3] with the six zeroed, +38.4% with random six; ordinal (no labels) +20.3% → +8.2% [7.2, 9.2] vs +20.5%; topic (no labels) +9.3% → +5.3% [4.5, 6.1] vs +9.2%. All six-vs-random p < 0.0001.
**Surprises.** (1) A task the model is told to do *after* answering ("after answering, summarize [Pk]") moves attention during the answer as much as an explicit attention instruction, and drives the six heads hardest of all. "Memorize it, I'll ask later" — also about the future — barely does anything. So the boost tracks "what I'll have to *process* next", not the word "attention" and not "this matters". (2) The six heads are causal for the boost under ordinal and topic addressing too (≈60% and ≈43% of it), where there is no label digit to copy: they are the address reader for instruction-directed attention in general. (3) "Do not think about [Pk]" raises attention to it (white-bear effect, small), while "ignore"/"skip" lower it a little.
**Next.** E4 (running): conflicting sources, does instructed attention choose the answer.
