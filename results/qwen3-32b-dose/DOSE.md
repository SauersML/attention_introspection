# Dose-response of instruction wording on native attention (Qwen3-32B, arithmetic task)

Code: `dose/dose.py` (pod), `dose/dose_analyze.py` (tables), `dose/dose_fig.py` (figure), `dose/dose_run.sh`.
Results: `qwen3-32b-dose/` (dose.json per-context records, dose_results.json, dose_tables.md, meta.json). Figure: `compute/fig/attn_intro/dose.png/.jpg`.
Pod: hypha-prod-attndose32 (A100-SXM4-80GB, final run 0.49 h, $1.06; all attempts incl. five failed launches $3.75 from the rp-run cost files). No pods left.

## Setup
- 150 arithmetic contexts built exactly as in experiment B (same SQuAD filter, same generator, 4 paragraphs [P1]..[P4], "Unrelated question: what is x + y? Answer with just the number."), new seed 7. Pk random per context (k ≠ j as before), same Pk in every condition.
- One inserted line at B's position (between the paragraphs and the question). Wording (fixed before running; every rung says "paragraph [Pk]"; unlike B, no "While you answer," prefix):
  - 0 none
  - 1 mention: "Note that one of the paragraphs is labelled [Pk]."
  - 2 "Glance at paragraph [Pk]."
  - 3 "Focus your attention on paragraph [Pk]."
  - 4 "Focus strongly on paragraph [Pk]."
  - 5 "Focus very strongly and only on paragraph [Pk]. Ignore everything else."
  - 6 the line of 5 given twice: once before "Here are four paragraphs.", once after the question (none at B's position)
  - 7 "Pay a little less attention to paragraph [Pk]."
  - 8 "Do not pay attention to paragraph [Pk]."
  - 9 "Completely ignore paragraph [Pk]. Do not look at it at all."
- Teacher-forced: every condition scores the no-instruction greedy answer; paragraph tokens checked identical across conditions. Attention as in B: patched eager attention, head-averaged, summed over Pk's body tokens / its length, mean over answer-token queries, band layers 21–53 of 64. Per-context % change = 100·(mass_cond / mass_ref − 1); 95% bootstrap CI over contexts (10,000 resamples).
- Free (untouched) answers: they differ from the no-instruction answer in 0–5.3% of contexts (6 "twice" 5.3%, 5 4.7%, 2 and 4 1.3%, others ≤0.7%); free accuracy ≥94.7% in every condition, 100% with no instruction.

## Table (n = 150; band 21–53; mean over answer queries)

| condition | vs mention (1) | contexts > 0 | vs none (0) | contexts > 0 | vs mention, first answer query only |
|---|---|---|---|---|---|
| 0 none | −7.3% [−9.1, −5.6] | 24% | — | — | +8.0% [+6.9, +9.1] |
| 1 mention | — | — | +9.4% [+7.4, +11.4] | 76% | — |
| 2 glance at | +38.0% [+35.5, +40.4] | 100% | +50.9% [+47.1, +54.7] | 99% | +6.9% [+6.1, +7.7] |
| 3 focus your attention on | +47.3% [+44.5, +50.1] | 100% | +61.5% [+56.9, +66.1] | 100% | +6.4% [+5.5, +7.2] |
| 4 focus strongly on | +38.0% [+35.6, +40.5] | 100% | +51.3% [+47.1, +55.3] | 99% | +3.6% [+2.6, +4.6] |
| 5 very strongly and only (+ "Ignore everything else.") | +20.9% [+18.9, +23.0] | 100% | +32.5% [+29.0, +35.9] | 95% | −6.2% [−7.4, −4.9] |
| 6 the same, twice | +202.0% [+189.9, +214.3] | 100% | +233.7% [+216.6, +251.2] | 100% | +45.6% [+40.9, +50.2] |
| 7 a little less | +1.0% [+0.1, +1.9] | 53% | +10.5% [+8.3, +12.7] | 76% | −5.9% [−6.6, −5.1] |
| 8 do not pay attention | −0.8% [−1.6, +0.1] | 40% | +8.5% [+6.4, +10.6] | 74% | −2.9% [−3.6, −2.2] |
| 9 completely ignore | −14.3% [−15.1, −13.4] | 0% | −6.4% [−8.1, −4.8] | 33% | −10.3% [−11.1, −9.5] |

## Monotonicity
**Up-ladder (2→6): not monotone.** Means vs mention: +38, +47, +38, +21, +202%. Spearman of ladder order vs mean = +0.30 (bootstrap 95% CI [+0.10, +0.30]; mean per-context Spearman +0.21 [+0.19, +0.24]). Paired steps:
- glance → focus: +6.8% [+6.0, +7.6] (93% of contexts up)
- focus → focus strongly: −6.1% [−6.7, −5.4] (5% up)
- focus strongly → very strongly and only: −12.2% [−13.0, −11.4] (1% up)
- very strongly and only → twice: +149.9% [+140.4, +159.6] (100% up)

So stronger adverbs lower attention on Pk after "focus your attention on"; the one big jump is "twice", which also moves the line (one copy right after the question, i.e. just before the answer, and none at B's position). That rung confounds repetition with position, so it is not evidence that intensity words work.

**Down-ladder (7→9): monotone, but only the last rung does anything.** Means vs mention +1.0, −0.8, −14.3%; Spearman −1.00 (bootstrap CI [−1.00, −1.00]; per-context −0.89 [−0.92, −0.86]). Steps: little less → do not −1.7% [−2.2, −1.3]; do not → completely ignore −13.5% [−14.2, −12.9] (99% of contexts down). "A little less" and "do not pay attention" are at the level of merely naming the paragraph; "Completely ignore paragraph [Pk]. Do not look at it at all." is the only line that lowers attention below the mention (−14%) and below no instruction at all (−6.4%).

**Vs none:** naming Pk alone raises attention on it by 9.4%; every up-rung and "a little less"/"do not" stay above none; only "completely ignore" goes below none.

## Per layer (mean % change vs mention; dose_tables.md has every 4th layer, dose_results.json all 64)
Effects sit at layers ~24–48 and peak at layer 28: focus +242%, glance +184%, focus strongly +192%, very strongly +112%, twice +1156%; completely ignore −25% at 28, −29% at 52. Layers 0–20 change ≤ ±16% in every condition.

## Reading
- Instructed control is real and large in this wording (+38–47% for plain focus/glance over a matched mention, 100% of contexts), but it does not scale with how emphatic the words are: "strongly" and "very strongly and only … Ignore everything else" give less, not more. One untested guess: the word "Ignore" in rung 5 pulls in the suppressive direction (rung 9 shows "ignore" lowers attention).
- Suppression is weak: only an explicit "completely ignore … do not look at it" moves attention below a mention.
- What would overturn the up-ladder conclusion: a "twice" rung that keeps both copies at B's position (repetition without the move) giving a big rise. Not run.
- Null: rung vs mention on the same contexts with the same answer tokens; the mention itself vs none is +9.4%, the size of a pure naming effect. No shuffled-Pk null was run.

## Differences from B
B used "While you answer, focus your attention on [Pk]." (bare label); here "Focus your attention on paragraph [Pk]." B's 32B arithmetic focus vs mention was +26% (geometric mean); here +47% (arithmetic mean of % change), different seed and wording, so not a like-for-like comparison.
