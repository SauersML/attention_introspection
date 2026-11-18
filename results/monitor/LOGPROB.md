# Does Qwen3-32B's report of its own attention carry anything a transcript reader lacks?

![Qwen3-32B, 300 prompts: the self-report's rank correlation with true attention (0.61) is about the same as the same model reading the transcript (0.59); adding the self-report to the reader adds at most a few hundredths of rank correlation, largest when asked one turn later with a 0–10 rating](../../figures/monitor_logprob.png)

## The story

Qwen3-32B answers a question about four paragraphs. Then we ask where its attention went ("Which paragraph did you attend to most while writing your answer?", a 0–10 rating per paragraph, "which least?", or a free description), and read the probabilities of its answer instead of its sampled answer. We compare that report with the true attention of its answer tokens to each paragraph, and with the same model asked the same question about the conversation given to it as quoted text (the transcript reader).

1. **The self-report tracks true attention about as well as the transcript reader does.** Asked right after answering, the per-prompt rank correlation between report and true attention is 0.61 for the self-report and 0.59 for the reader; the difference has a confidence interval that includes zero. Most of what the self-report "knows" is what anyone reading the conversation could guess (the paragraph the question points to, the paragraph the answer quotes).
2. **There is a small signal on top of the reader, and it is real but tiny.** Adding the self-report's scores to a model that already has the reader's scores, paragraph position and length raises the held-out rank correlation by about 0.01–0.02 for most ways of asking, and by 0.04 when the model is asked one turn later with 0–10 ratings. Shuffling the self-report across prompts never gives this much (p < 0.005), so the extra information is specific to the prompt. But in all but one setting the gain's confidence interval over prompts includes zero: the effect is consistent in sign, not large on any single prompt.
3. **Showing the model examples of its true attention makes its self-report worse, not better.** After 4 or 16 earlier conversations with their measured attention, the reader improves (0.63) while the self-report drops (0.47 for "which most?", 0.22 and 0.16 for 0–10 ratings), and the self-specific gain disappears (p = 0.74, 0.04, 1.00, 0.61). The examples seem to make the self-report copy the examples' pattern instead of consulting anything about this answer.

## Method

- Data: `hypha-prod-mon{0,1,2}/out/hf1.json` and `hf2.json` (feedback reads), 300 prompts in five families of 60 (results/monitor/PREREG.md describes the prompts, wordings and reader).
- True attention per paragraph: attention mass from the answer tokens to the paragraph's tokens (averaged over heads), averaged over layers 21–53. With thinking, the final answer's attention (`truth_think.final`). The feedback rows ask right after answering, so their truth is the normal answer's.
- Report score per paragraph: log probability of the label for "which most?" and the free answer (the free description parsed by the same model), minus log probability for "which least?", the expected 0–10 rating for ratings. Reader scores are built the same way.
- (1) Per prompt, Spearman correlation of the four report scores with the four true attention values; mean over prompts with a 10,000-resample bootstrap CI.
- (2) Self-specific gain: within each prompt, report scores and true attention are turned into ranks and z-scored. A linear model predicts the true-attention rank of each paragraph from the reader's rank, paragraph position (one-hot) and log token count (centred within the prompt); a second model adds the self-report's rank. Both are fitted with 10-fold cross-validation grouped by prompt; the held-out predictions give a within-prompt Spearman per prompt, and the gain is its mean difference. Because one 10-fold split moves the gain by about ±0.01 (the size of the effect), the split is redrawn 10 times and per-prompt results averaged. CI: 200 bootstrap resamples of prompts. Null: the self-report's ranks shuffled across prompts within each paragraph slot (200 shuffles, each with the same 10 splits); p = share of shuffles with a gain at least the observed one.
- Code: `monitor/logprob.py` (numbers, writes results/monitor/logprob_results.json and prints the tables below), `figures/monitor_logprob_fig.py` (figure). Example prompt in the figure: id 173 (rank correlation 0.80, a typical value).

## (1) Rank correlation with true attention, mean over 300 prompts [95% CI]

| way of asking | self-report | same model reading the transcript | self − reader |
|---|---|---|---|
| most attended, asked right after | 0.615 [0.578, 0.651] | 0.592 [0.552, 0.632] | +0.023 [-0.003, +0.050] |
| most attended, asked before answering | 0.587 [0.548, 0.627] | 0.567 [0.528, 0.605] | +0.020 [-0.007, +0.047] |
| most attended, asked one turn later | 0.427 [0.373, 0.481] | 0.449 [0.395, 0.500] | -0.022 [-0.069, +0.024] |
| most attended, with thinking | 0.484 [0.432, 0.534] | 0.475 [0.423, 0.527] | +0.009 [-0.023, +0.040] |
| most attended, after 4 true examples | 0.467 [0.414, 0.519] | 0.628 [0.591, 0.665] | -0.161 [-0.214, -0.108] |
| most attended, after 16 true examples | 0.447 [0.395, 0.498] | 0.618 [0.579, 0.656] | -0.170 [-0.225, -0.116] |
| 0–10 rating, asked right after | 0.583 [0.542, 0.623] | 0.615 [0.576, 0.651] | -0.032 [-0.071, +0.007] |
| 0–10 rating, asked before answering | 0.563 [0.519, 0.605] | 0.594 [0.552, 0.635] | -0.031 [-0.056, -0.007] |
| 0–10 rating, asked one turn later | 0.420 [0.364, 0.474] | 0.427 [0.371, 0.481] | -0.007 [-0.063, +0.050] |
| 0–10 rating, after 4 true examples | 0.222 [0.159, 0.285] | 0.596 [0.556, 0.634] | -0.374 [-0.442, -0.308] |
| 0–10 rating, after 16 true examples | 0.158 [0.097, 0.219] | 0.597 [0.558, 0.636] | -0.439 [-0.510, -0.369] |
| least attended, asked right after | 0.477 [0.426, 0.529] | 0.492 [0.440, 0.542] | -0.015 [-0.051, +0.023] |
| free answer, asked right after | 0.498 [0.450, 0.546] | 0.496 [0.444, 0.546] | +0.002 [-0.039, +0.044] |

## (2) What the self-report adds over the transcript reader (held-out within-prompt rank correlation)

| way of asking | base (reader + position + length) | + self-report | gain [95% CI] | shuffle null 95th pct | p |
|---|---|---|---|---|---|
| most attended, asked right after | 0.690 | 0.699 | +0.009 [-0.007, +0.028] | +0.002 | < 0.005 |
| most attended, asked before answering | 0.670 | 0.675 | +0.005 [-0.014, +0.023] | +0.003 | 0.030 |
| most attended, asked one turn later | 0.615 | 0.635 | +0.020 [-0.000, +0.039] | +0.003 | < 0.005 |
| most attended, with thinking | 0.590 | 0.602 | +0.013 [-0.006, +0.032] | +0.002 | < 0.005 |
| most attended, after 4 true examples | 0.699 | 0.697 | -0.001 [-0.015, +0.016] | +0.004 | 0.740 |
| most attended, after 16 true examples | 0.674 | 0.678 | +0.003 [-0.010, +0.016] | +0.003 | 0.040 |
| 0–10 rating, asked right after | 0.687 | 0.690 | +0.003 [-0.016, +0.023] | +0.003 | 0.050 |
| 0–10 rating, asked before answering | 0.664 | 0.664 | -0.000 [-0.009, +0.007] | +0.002 | 0.405 |
| 0–10 rating, asked one turn later | 0.624 | 0.664 | +0.040 [+0.013, +0.068] | +0.005 | < 0.005 |
| 0–10 rating, after 4 true examples | 0.692 | 0.686 | -0.006 [-0.017, +0.003] | +0.003 | 1.000 |
| 0–10 rating, after 16 true examples | 0.687 | 0.686 | -0.001 [-0.010, +0.006] | +0.003 | 0.615 |
| least attended, asked right after | 0.602 | 0.618 | +0.016 [-0.004, +0.037] | +0.006 | < 0.005 |
| free answer, asked right after | 0.662 | 0.680 | +0.018 [-0.006, +0.039] | +0.002 | < 0.005 |

## Reading the null and the CI together

The two answer different questions. The shuffle null asks whether the self-report carries prompt-specific information beyond the reader: a gain this large never came from shuffled self-reports (p < 0.005 in 6 of 13 settings). The bootstrap CI asks how large and how consistent the gain is across prompts: it includes zero in 12 of 13 settings, because the gain is a sum of many prompts where the self-report changes nothing and a few where it helps or hurts. The honest summary is "a real but very small private signal", largest (+0.04) one turn after the answer with 0–10 ratings.

## Checks and caveats

- What would overturn point 2: the gain depending on one fold split. A single split moves it by ±0.01, so every number here is the mean over 10 splits, and the null uses the same 10 splits.
- True attention here is total attention mass. With the preregistered length-normalised attention (mass divided by token count) both self and reader correlate less (0.42 and 0.43 asked right after, most attended) and the self-report is again no better than the reader in any setting; after true examples the self-report again falls below the reader (0.40 vs 0.44 and 0.38 vs 0.45 for most attended; 0.18 vs 0.45 and 0.17 vs 0.44 for ratings).
- The feedback examples showed length-normalised percentages, while the main truth here is total mass; the self-report drop after examples appears under both truths.
- Rank correlations over four paragraphs take few distinct values, hence the bands of dots in the figure.
