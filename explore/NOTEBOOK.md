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

## E4 — Conflicting sources: does instructed attention choose the answer? (05:40–06:00)
**Why.** E2: when one paragraph holds the answer, the instruction barely moves attention and doesn't change answers. With two paragraphs that *disagree*, the model has to pick a source, so the instruction has room to matter.
**Setup.** 150 SQuAD paragraphs whose answer is a 1–4 digit number appearing once (seed 23). A copy of the paragraph with that number replaced (years ±7–60, other numbers by 25–50%), plus two unrelated paragraphs, shuffled, labelled [P1]–[P4]. Question: "Question: … Answer with just the answer." Conditions: none, and four lines each naming the original or the altered paragraph: focus ("Focus your attention on paragraph [Pj]."), use ("Answer using only paragraph [Pj]."), ignore ("Completely ignore paragraph [Pj]."), mention. Readout: greedy answer (contains original number only = 1, altered only = 0, both = 0.5) and teacher-forced log p(original) − log p(altered) in nats. e4_conflict.py, a4.py → out/e4/a4.json.
**Control.** Mention naming the same paragraph; the "swing" = (named original) − (named altered).
**Result.** With no instruction the model gives the original number 75% [68, 81] of the time (it has seen the real fact; also earlier paragraph wins slightly more).
| line naming the paragraph | answer from the original, naming original / naming altered | swing in answer choice |
|---|---|---|
| mention | 0.85 / 0.58 | +0.27 [0.20, 0.34] |
| Focus your attention on … | 0.98 / 0.06 | +0.93 [0.88, 0.97] |
| Answer using only … | 0.97 / 0.02 | +0.95 [0.92, 0.98] |
| Completely ignore … | 0.19 / 0.98 | −0.79 [−0.85, −0.72] |
Log-prob swing minus mention's: focus +23.3 nats [22.0, 24.5], use +28.2 [26.9, 29.5], ignore −23.9 [−25.6, −22.1]; all p < 0.0001.
Attention (band mean, vs none): focus on the altered copy +19.5% on it, −14.1% on the original. **Ignore the original: −8.1% on the original, −4.7% on the altered one** — almost no relative shift, yet the answer flips to the altered number 81% of the time.
**Surprises.** (1) "Focus your attention on [Pj]" works as a source selector almost exactly like "answer using only [Pj]" — it overrides the model's own knowledge of the true fact 94% of the time. (2) "Ignore" barely moves attention (E1/E3: −6%, mostly not specific) but is a strong behavioural filter. So for ignore, the instruction acts after reading: the paragraph is still attended, its content is discarded downstream. Focus may work the same way, with the attention rise a side effect rather than the cause.
**Next.** E6: where is the selection? Record attention to the two conflicting *number tokens* (the only tokens that differ) per head, under focus/ignore/use; then causal test: under "focus [Palt]", block the answer's attention to the altered paragraph's number: does the model fall back to the original (attention carries the choice) or still say the altered number (choice made earlier, at the number via the paragraph's own representation)?

## E5 — Graded control: "X% on [Pa], (100−X)% on [Pb]" (06:00–06:10)
**Why.** E3/E4 show the instruction acts like a pointer. A pointer is binary; can the model also take a *ratio*? (Open direction from the prior-work search; nothing found on it.)
**Setup.** 150 contexts (seed 11), labelled paragraphs, unrelated sum, no-instruction answer teacher-forced. Line: "Split your attention: X% on paragraph [Pa] and (100−X)% on paragraph [Pb]." for X = 100, 90, 75, 50, 25, 10, 0, and the same allocation with the two paragraphs named in the other order ("(100−X)% on [Pb] and X% on [Pa]"). Also words: "Focus mostly on [Pa] and a little on [Pb]" and the reverse. e5_graded.py, a5.py → out/e5/a5.json.
**Control.** "Note that two of the paragraphs are labelled [Pa] and [Pb]." Metric: log(attention a / attention b), each relative to that control.
**Result.**
- Every split line *lowers* attention to both named paragraphs by ~10% vs the mention line; the share modulates mostly the paragraph given less: X = 100: a −5%, b −15%; X = 0: a −13%, b −2%.
- The ratio follows the number monotonically: per-prompt Spearman over the seven levels 0.86 [0.84, 0.88] (and 0.85 with the naming order swapped). But compressed: 100:0 vs 0:100 is a 25% change in the a/b ratio (log ratio +0.11 vs −0.12); 90:10 vs 10:90 only 6% [5.2, 6.7]; 75:25 vs 25:75 3.9% [3.3, 4.4]. Plain "Focus on [Pk]" moves its paragraph +47%.
- Naming order doesn't matter for numbers (50/50 in either order: difference −0.3% [−1.4, +0.8]), but it does for words: "mostly [Pa], a little [Pb]" +18% for a over b; "a little [Pa], mostly [Pb]" still +7% for a — the first-named paragraph wins regardless of which word goes with it.
**Surprises.** The model reads the percentage and applies it to the right paragraph whatever the naming order, with high per-prompt consistency, yet the effect is tiny: graded control exists but its gain is ~1/10 of a plain pointer, and "split your attention" as a frame suppresses both paragraphs. Words do worse than numbers: "a little / mostly" is overridden by mention order. The curve (figures/e5_graded_split.png) is a step at the ends with a shallow slope between: 0% and 100% behave like categorical ignore/attend (±11%), 10%→90% spans only ~7%.
**Next.** E6 (running): where the source is selected in the conflict task.

## E6 — Where is the source chosen? (06:10–06:40)
**Why.** E4: "focus" and "ignore" decide which conflicting number the model gives, but whole-paragraph attention (layers 21–53, all answer tokens) barely separated the two copies under ignore. Either the choice is made without attention at answer time, or my metric looks in the wrong place.
**Setup.** The same conflict sets as E4 (seed 23; 120 items, 1 dropped where the altered number occurs twice → n = 120). (a) Attention from the last prompt token (the position that predicts the first answer token) to the conflicting number's tokens in each copy, per layer and head; (b) causal: in every layer, block attention (logit −10⁴) from the answer-predicting positions to the original's or the altered copy's number tokens, or to its whole paragraph; readout log p(original) − log p(altered), both teacher-forced. e6_select.py, a6.py → out/e6/a6.json.
**Control.** No instruction; for (a) every condition vs none, and named-altered vs named-original.
**Result.**
- Last-position attention to the number tokens (all heads, layers 21–53, vs none): focus on the altered copy: altered number +67% [57, 78], original −46% [−49, −43]. **Ignore the original: original number −37% [−40, −34], altered +43% [37, 51].** Use-only the altered copy: +49% / −55%. Mention alone: both −12 to −28%.
- By layer, the selection (log of altered/original attention, named-altered minus named-original) is exactly zero up to layer 43 and then large in layers 45–62, peaking at layer 52: 3.6 nats for focus, 3.5 for use-only, 2.3 for ignore. Halved, that is the chosen number's attention ÷ the other number's, geometric mean over the two namings (cancels any baseline preference): ≈6× for focus and use-only, ≈3× for ignore at layer 52.
- Blocking (margin in nats): no instruction +4.6; block the altered number → +20.5, block the original number → −10.3. Under "focus on the altered copy" (−11.9), blocking the altered number flips it to +15.0 [13.2, 16.6]; under "ignore the original" (−6.5), blocking the altered number gives +20.1, the same as with no instruction. Blocking the whole paragraph ≈ blocking just its number.
**Surprises.** (1) The source choice is carried out by attention after all — but by late layers (45–62) at the answer position onto the specific answer tokens, an up to ~6-fold selection, invisible in the earlier whole-paragraph metric. (2) "Ignore" works the same way as "focus": it reroutes late attention away from the ignored paragraph's answer token, though it hardly changes attention to the paragraph as a whole. (3) Cut the late attention route and the instruction is nearly powerless: under "ignore the original", blocking the altered number gives the original answer as firmly as with no instruction at all. So the instruction does not pre-compute the answer elsewhere; it only steers which tokens late heads read. Picture: layers 20–26 read the instruction's address (six heads, E3); layers 45–62 execute it at the point of reading the answer.
**Next.** Which heads do the late selection, and do they take their cue from the six early heads? E7: zero the six early heads in the conflict task: does the late selection (and the answer choice) collapse?

## E7 — Does the early instruction reader drive the late number readers? (started 06:50)
**Why.** E6 points to two stages: six heads in layers 20–26 read the instruction's address (E3), and late heads (45–62) at the answer position pick which paragraph's number to read. The late heads are concentrated: in E6, head 52.36 puts 72% of its last-position attention on the two numbers and its selection under focus is a ratio-of-ratios of 8.6 nats; the 15 heads with the highest selectivity × attention mass are in layers 50–59 (out/e6/late_heads.json). Focus and ignore use the same late heads (selectivity correlates 0.95 across late heads).
**Setup.** Fresh items: the next 120 conflict sets after E6's (same seed-23 stream). Ablations (zero outputs at all positions): the six early heads vs a random six from layers 20–26; the 15 late heads (chosen on E6's items) vs a random 15 from the same layers. Conditions: none, focus orig/alt, ignore orig/alt. Readout: margin, and late selection (layers 45–62) under early ablation. e7_chain.py, a7.py.
(Note: the first E7 run crashed at the end — the numeric-answer pool has only 195 items, and I saved only at the end; rerun at 07:00 with saves every 10 items. n = 75 fresh items, 120–194.)
**Control.** Random six early heads; random 15 late heads from the same layers.
**Result** (swing = margin when the line names the original minus when it names the altered copy, nats; intact focus +28.5 [26.5, 30.3], ignore +18.1 [16.1, 20.0]):
- **Zeroing the six early heads changes nothing:** focus swing +0.14 [−0.08, +0.37], ignore −0.28 [−0.52, −0.03]; random six +0.13 / −0.02. The late number selection (layers 45–62) is also unchanged: focus 1.73 nats with the six zeroed vs 1.69 intact.
- Zeroing the 15 late heads cuts the swing by 3.8 [3.1, 4.5] nats for focus (−13%) and 3.3 [2.4, 4.1] for ignore (−18%); random late 15: −0.3 / −0.4. With them off, "ignore the original" still flips the answer 73% of the time (vs 81%).
**Surprises.** The six heads that carry ~60% of the paragraph-wide attention boost in the sum task (E3) play no part in actually following the instruction when it matters. The attention boost that this whole line of work measured (and that the model's self-reports were compared against) is dissociable from the instruction's behavioural effect: one is a by-product read in layers 20–26, the other is executed in layers 45–62 by many redundant heads (the top 15 carry under a fifth).
**Next.** E8 (running): residual patching of the choice between the two prompts that differ in one digit, by layer and position.
