# Question-boost experiment, labels in order (Qwen3-32B)

Rerun of arm C of `results/privileged_shuffled` (attention onto one paragraph boosted at the report-question queries only, layers 21-53,
the answer's KV cache unboosted), with the labels [P1]..[P4] in slots 1..4 and the report options listed in order.
Same 100 paragraph prompts (60 arithmetic, 40 reading-question), same seed, report wording W1. Code: `privileged_inorder/`.
Files: `data/recs_0.jsonl` (100 records, arm C at multipliers 5 and 10 plus no injection), `results.json`, `analyze.log`, `meta.json`.
The shuffled arm is the original run's records (`results/privileged_shuffled/data`), compared on the same 100 prompts.

Checks: per-paragraph token counts match the shuffled run on 100/100 prompts; 91/100 answers identical; the answer is correct on
100% of arithmetic and 70% of reading-question prompts; measured attention onto the boosted paragraph x3.8 (m=5) and x6.2 (m=10),
x3.8 and x6.3 in the shuffled run.

## Probability of the boosted paragraph (the existing metric)
Per prompt: change in log-probability of the boosted paragraph's label vs no injection, minus the mean change of the other three,
averaged over the four rotations; shown as the percent increase in probability (exp(mean) - 1). 95% CI by bootstrap over prompts,
p by sign-flip test.

| | in-order | shuffled | in-order minus shuffled |
|---|---|---|---|
| m=10, all 100 | +97% (p < 1e-5) | +64% (p < 1e-5) | +20%, p = 0.005 |
| m=10, arithmetic (60) | +135% (p < 1e-5) | +112% (p < 1e-5) | +11%, p = 0.27 |
| m=10, reading question (40) | +51% (p = 9e-5) | +12% (p = 0.26) | +35%, p = 0.0003 |
| m=5, all 100 | +21% (p = 1.5e-5) | +18% (p = 1.5e-5) | +3%, p = 0.53 |
| m=5, arithmetic (60) | +21% (p = 0.0006) | +27% (p = 3e-5) | -4%, p = 0.44 |
| m=5, reading question (40) | +21% (p = 0.0002) | +6% (p = 0.19) | +14%, p = 0.0095 |

The in-order gain at m=10 comes from the reading-question prompts. No-injection mean probability by slot, in-order: .22 .21 .25 .33
(shuffled: .23 .21 .28 .29), so in order Qwen leans toward the last paragraph without injection.

## Top answer (figures/hero_question_top.png)
Per prompt: percent of the four rotations whose top (argmax) label is the boosted paragraph, minus 25 points (the no-injection top
answer matches one rotation in four).

| | in-order | shuffled (same prompts) | in-order minus shuffled |
|---|---|---|---|
| m=10 | +5.8 points [+3.2, +8.5], p = 2e-5 | +5.5 [+3.2, +8.0], p = 2e-5 | +0.2 [-3.0, +3.5], p = 1 |
| m=5 | +0.5 [-1.0, +2.0], p = 0.75 | +2.0 [+0.2, +3.8], p = 0.06 | -1.5 [-3.8, +0.8], p = 0.29 |

So in-order labels raise the boosted label's probability but do not change how often it becomes the top answer: at m=10, 31% of
top answers are the boosted paragraph vs 25% without injection, in both label orders. Per-prompt values at m=10 (in-order): -25: 2,
0: 76, +25: 19, +50: 3. Figure example: prompt 6 (arithmetic, 87 + 83), top answer P2 with paragraph 2 boosted, value 0 = the
median; its no-injection top answer is also P2.
