# Privileged channel and attention instructions on a list of single words (Qwen3-32B), labels and options in order

A follow-up to the privileged-channel experiment (`results/privileged_inorder_full/`; plan in `results/privileged_shuffled/PREREG.md`) with the same design on a simpler context: a list of 8 single-token nouns, then an unrelated arithmetic question. Not separately preregistered; one wording (W1) and one setup, so that experiment's rules R3 and R4 do not apply.

This is the in-order run: labels [W1]..[W8] in slots 1..8 and the report options listed in order. The first run, identical except that the labels were randomly permuted to positions and the options listed in a random order, is `results/privileged_words_shuffled/`; both side by side, paired by context, in `compare_shuffled.md` (`privileged/compare_words.py`).

Code: `privileged/privileged_words.py --inorder` via `privileged/run_words_inorder.sh` (two A100-80GB pods, 50 contexts each, the same shard split as the first run), `privileged/analyze_words.py` (tables). Data: `data/recs_{0,1}.jsonl.gz` (gunzip before re-running the analysis), `data/run_{0,1}.log`, `results.json`, `tables.md`.

Checks: same 100 contexts (same words, seed 11); answers identical to the shuffled run in 100 of 100; arithmetic correct in 99 of 100.

## Setup
- 100 contexts. "Here is a list of eight words." then eight lines "[Wl] noun", one noun from each of 8 different semantic categories (no two related), each a single token; labels [W1]..[W8] in order; then "Unrelated question: what is X + Y? Answer with just the number."
- The answer is generated once and teacher-forced in every condition (identical text).
- Boost: log(multiplier) added to the pre-softmax attention scores on one word's token, all heads, layers 21–53 of 64; ×¼ ×½ ×2 ×5 ×10; each of the 8 words in turn.
- Arms: A, only the answer queries (the report question's own pass unmodified; the only route is the KV cache at the answer positions); B, answer queries and report-question queries; C, report-question queries only.
- Report: "Which word did you attend to most while writing your answer? Answer with just one of [W1], [W2], [W3], [W4], [W5], [W6], [W7], or [W8].", prefilled "[W", log-probabilities over the 8 label digits.
- Specificity = change in log P(report = boosted word) minus the mean change for the other seven words, averaged over the rotation; 95% bootstrap CIs over contexts. "%" = exp(specificity) − 1. Null: any perturbation that does not depend on which word was boosted gives 0.

## Privileged channel

Manipulation check (attention mass on the boosted word, ratio to no boost): ×5 gives 4.62× at the answer queries and 4.17× at the report-question queries; ×10 gives 8.60× and 7.46×.

| multiplier | A: answer only | B: both | C: question only |
|---|---|---|---|
| ×¼ | +0.012 [+0.004, +0.019] (+1%) | −0.084 [−0.119, −0.052] (−8%) | −0.092 [−0.129, −0.059] (−9%) |
| ×½ | +0.001 [−0.008, +0.009] (+0%) | −0.078 [−0.107, −0.053] (−8%) | −0.075 [−0.102, −0.052] (−7%) |
| ×2 | −0.003 [−0.011, +0.004] (−0%) | +0.220 [+0.173, +0.273] (+25%) | +0.216 [+0.168, +0.272] (+24%) |
| ×5 | +0.003 [−0.006, +0.013] (+0%) | +0.906 [+0.803, +1.014] (+148%) | +0.905 [+0.802, +1.011] (+147%) |
| ×10 | +0.032 [+0.022, +0.042] (+3%) | +1.940 [+1.800, +2.085] (+596%) | +1.917 [+1.780, +2.054] (+580%) |

Dose (mean per-context Spearman of log multiplier vs specificity): A +0.07 [−0.02, +0.17]; B +0.87 [+0.83, +0.91]; C +0.87 [+0.82, +0.91].

Against the shuffled run (paired, same contexts): arm A is about the same (×10: +3.2% against +1.2%; ×¼: +1.2% against −0.9%; ×½ to ×5 within ±1% in both). Arm C is far larger in order: ×10 +580% against +25%, ×5 +147% against +12%, and the low doses now go below zero (×¼ −9%, ×½ −7%; shuffled −1% and −1%, intervals including 0). Arm B tracks arm C in both runs. In the shuffled run the boost did not move the report toward the label number equal to the boosted word's position (−1% at ×10), nor toward the label listed at that position among the options (−2%), so the shuffled run's smaller effect is not a mix-up of label and position.

Same task with four paragraphs instead of eight words (`results/privileged_inorder_full/`, paragraphs-arithmetic, 60 contexts, in order):

| | words, ×5 | words, ×10 | paragraphs, ×5 | paragraphs, ×10 |
|---|---|---|---|---|
| A: answer only | +0% | +3% | −7% | −14% |
| C: question only | +147% | +580% | +21% | +132% |

## Instruction test
One line placed just before the arithmetic question, for each word k in turn: "Focus your attention on word [Wk].", "Completely ignore word [Wk]. Do not look at it at all.", or the control "Note that one of the words is labelled [Wk]." (and no line). Attention mass on word k's token at the answer-token queries (teacher-forced on the no-instruction answer), head mean; ratio per context, averaged over k; "right way" = share of (context, word) pairs where the named word's attention moved in the instructed direction.

| instruction | vs | layers 21–53 | layer 28 | right way (21–53) |
|---|---|---|---|---|
| focus | mention | +34.5% [+33.5, +35.6] | +123.8% [+118.3, +129.3] | 99% |
| ignore | mention | −12.1% [−12.7, −11.5] | −20.8% [−22.2, −19.4] | 80% |
| focus | none | +18.3% [+17.0, +19.6] | +105.3% [+99.0, +111.7] | 89% |
| ignore | none | −22.7% [−23.3, −22.1] | −27.4% [−29.3, −25.4] | 98% |
| mention | none | −12.1% [−12.8, −11.4] | −8.3% [−10.6, −5.8] | (86% down) |

Against the mention line, the named word's change minus the other seven words' change:

| instruction | named minus others, 21–53 | layer 28 | other words' own change, 21–53 | layer 28 |
|---|---|---|---|---|
| focus | +6.4% [+5.9, +6.8] | +27.6% [+25.7, +29.5] | +26.5% [+25.7, +27.3] | +75.4% [+71.8, +79.1] |
| ignore | −6.2% [−6.5, −5.9] | −6.8% [−7.2, −6.5] | −6.4% [−6.9, −5.8] | −15.0% [−16.5, −13.5] |

The named word is the single word whose attention moves most in 21% (focus) and 19% (ignore) of cases at layers 21–53 (31% / 19% at layer 28), against 12.5% by chance. The instruction line leaves the free answer unchanged in 96–99% of cases.

## Reading
- On single words there is still no answer-time trace like the paragraphs' inverted one: arm A stays within ±1% from ×½ to ×5 and has no dose trend; at ×10 it is +3% (interval +2% to +4%), small and the opposite sign to paragraphs (−14%). The boost raised attention mass on the word 4.6× at ×5 and 8.6× at ×10, so the manipulation worked.
- Steering attention while the model reads the report question (arm C) moves the report strongly toward the boosted word, with dose (+147% at ×5, +580% at ×10), more than for paragraphs (+132% at ×10). With labels in order the report follows the attention used to answer the report question much more than the shuffled run showed (+25% at ×10).
- "Focus on word [Wk]" mostly raises attention to every word in the list (+27% over the mention line in layers 21–53, +75% at layer 28) and the named word by a further 6% (28% at layer 28). "Ignore word [Wk]" lowers every word by 6% (15% at layer 28) and the named word by a further 6%. "Mention" itself lowers attention to the words by 12% against no line, so the mention line is the matched control.
