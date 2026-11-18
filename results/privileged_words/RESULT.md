# Privileged channel and attention instructions on a list of single words (Qwen3-32B)

A follow-up to `results/privileged/` with the same design (PREREG.md there), on a simpler context: a list of 8 single-token nouns, then an unrelated arithmetic question. Not separately preregistered; one wording (W1) and one setup, so that experiment's rules R3 and R4 do not apply.

Code: `privileged/privileged_words.py` (pod), `privileged/analyze_words.py` (tables; the same statistics as `analyze_priv.py`, plus the instruction test). Data: `data/recs_{0,1}.jsonl.gz` (gunzip before re-running the analysis), `results.json`, `tables.md`.

## Setup
- 100 contexts (two pods, 50 each). "Here is a list of eight words." then eight lines "[Wl] noun", one noun from each of 8 different semantic categories (no two related), each a single token; labels randomly permuted to positions; then "Unrelated question: what is X + Y? Answer with just the number." Arithmetic correct in 99 of 100.
- The answer is generated once and teacher-forced in every condition (identical text).
- Boost: log(multiplier) added to the pre-softmax attention scores on one word's token, all heads, layers 21–53 of 64; ×¼ ×½ ×2 ×5 ×10; each of the 8 words in turn.
- Arms: A, only the answer queries (the report question's own pass unmodified; the only route is the KV cache at the answer positions); B, answer queries and report-question queries; C, report-question queries only.
- Report: "Which word did you attend to most while writing your answer? Answer with just one of [W..], …", prefilled "[W", log-probabilities over the 8 label digits.
- Specificity = change in log P(report = boosted word) minus the mean change for the other seven words, averaged over the rotation; 95% bootstrap CIs over contexts. "%" = exp(specificity) − 1. Null: any perturbation that does not depend on which word was boosted gives 0.

## Privileged channel

Manipulation check (attention mass on the boosted word, ratio to no boost): ×5 gives 4.63× at the answer queries and 4.21× at the report-question queries; ×10 gives 8.62× and 7.56×.

| multiplier | A: answer only | B: both | C: question only |
|---|---|---|---|
| ×¼ | −0.009 [−0.018, −0.000] (−1%) | −0.014 [−0.039, +0.011] (−1%) | −0.008 [−0.033, +0.017] (−1%) |
| ×½ | +0.004 [−0.005, +0.012] (+0%) | −0.010 [−0.026, +0.006] (−1%) | −0.012 [−0.029, +0.005] (−1%) |
| ×2 | +0.002 [−0.006, +0.010] (+0%) | +0.044 [+0.012, +0.080] (+4%) | +0.051 [+0.018, +0.087] (+5%) |
| ×5 | −0.004 [−0.014, +0.005] (−0%) | +0.112 [+0.049, +0.177] (+12%) | +0.112 [+0.052, +0.174] (+12%) |
| ×10 | +0.012 [+0.002, +0.022] (+1%) | +0.237 [+0.159, +0.318] (+27%) | +0.220 [+0.141, +0.303] (+25%) |

Dose (mean per-context Spearman of log multiplier vs specificity): A +0.06 [−0.03, +0.15]; B +0.29 [+0.14, +0.44]; C +0.23 [+0.08, +0.38].

Same task with four paragraphs instead of eight words (`results/privileged/`, paragraphs-arithmetic, 60 contexts):

| | words, ×5 | words, ×10 | paragraphs, ×5 | paragraphs, ×10 |
|---|---|---|---|---|
| A: answer only | −0% | +1% | −7% | −13% |
| C: question only | +12% | +25% | +27% | +112% |
| Spearman A / C | +0.06 / +0.23 | | −0.60 / +0.47 | |

## Instruction test
One line placed just before the arithmetic question, for each word k in turn: "Focus your attention on word [Wk].", "Completely ignore word [Wk]. Do not look at it at all.", or the control "Note that one of the words is labelled [Wk]." (and no line). Attention mass on word k's token at the answer-token queries (teacher-forced on the no-instruction answer), head mean; ratio per context, averaged over k; "right way" = share of (context, word) pairs where the named word's attention moved in the instructed direction.

| instruction | vs | layers 21–53 | layer 28 | right way (21–53) |
|---|---|---|---|---|
| focus | mention | +30.6% [+29.6, +31.6] | +93.1% [+88.7, +97.5] | 99% |
| ignore | mention | −9.0% [−9.6, −8.4] | −16.5% [−18.0, −14.9] | 71% |
| focus | none | +11.9% [+10.8, +12.9] | +71.1% [+66.5, +75.8] | 78% |
| ignore | none | −22.0% [−22.6, −21.4] | −26.0% [−27.9, −24.0] | 96% |
| mention | none | −14.3% [−15.0, −13.8] | −11.4% [−13.3, −9.3] | (89% down) |

Most of the change is not about the named word. Against the mention line, the named word's change minus the other seven words' change:

| instruction | named minus others, 21–53 | layer 28 | other words' own change, 21–53 | layer 28 |
|---|---|---|---|---|
| focus | +3.3% [+3.0, +3.6] | +12.1% [+11.1, +13.2] | +26.4% [+25.5, +27.3] | +72.2% [+68.5, +76.0] |
| ignore | −4.4% [−4.6, −4.2] | −3.9% [−4.2, −3.5] | −4.8% [−5.4, −4.2] | −13.1% [−14.6, −11.6] |

Per context (averaged over the 8 words), the named-word specificity has the instructed sign in 100 of 100 contexts for focus and for ignore at layers 21–53 (98 of 100 for ignore at layer 28). The named word is the single word whose attention moves most in only 16% of cases (focus and ignore, layers 21–53; 22% / 17% at layer 28), against 12.5% by chance. The instruction line leaves the free answer unchanged in 97–99% of cases.

## Reading
- On single words, the answer-time trace that the paragraph experiment found is gone: arm A is within ±1% at every dose and has no dose trend, although the boost raised attention mass on the word 4.6× at ×5 (the manipulation worked). The answer is the same 2–3 digit tokens as in paragraphs-arithmetic, where arm A gave −7% at ×5, so the difference is the context: boosting one token's key leaves no specific trace the report can read; boosting a paragraph's 60–110 tokens did.
- Steering attention while the model reads the report question (arm C) still moves the report toward the boosted word, with dose, but about four times less at ×10 than for paragraphs (+25% vs +112%). The report follows the attention used to answer the report question, weakly here.
- "Focus on word [Wk]" mostly raises attention to every word in the list (+26% over the mention line in layers 21–53, +72% at layer 28) and the named word by a further 3% (12% at layer 28). "Ignore word [Wk]" lowers every word by 5% (13% at layer 28) and the named word by a further 4%. The model's attention follows the instruction's target consistently but slightly; the large effect is on the list as a whole. "Mention" itself lowers attention to the words by 14% against no line (the added line competes for attention), so comparisons with no line mix in that dilution; the mention line is the matched control.

What would overturn "no answer-time trace for single words": a trace on positions other than the word token (e.g. the label tokens) that the rotation would still see; not tested. The arm A ×¼ and ×10 CIs each just exclude 0 (−1%, +1%), in opposite directions to the paragraph trace; with 15 cells tested, I read these as noise, not a reversed trace.

## Caveats
- Cached-path check: for 4 contexts the no-boost report from the cached path differed from one uncached forward by up to 0.62 in log-probability (maximum over labels, bf16; 0.44 in the paragraph run). Every condition uses the same cached path, so the comparisons share it, but this is larger than every arm A cell.
- With no boost, the report favours the first list position (24% vs 12.5% uniform) and is fairly peaked (mean top label probability 0.68).
- The boost is a uniform bias over all heads in 33 layers, outside normal operation.

## Cost
RunPod, from the runner log: $2.59 for the two A100-80GB pods that produced the data (0.50 h, 0.69 h), $0.93 for two first launches that failed at setup (a script path missing on the pod), $0.02 for the 0.6B smoke test; $3.54 in all. All pods deleted (confirmed in the runner log; none named hypha-prod-privwords in rp-run status).
