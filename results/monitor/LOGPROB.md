# Does Qwen3-32B's report of its own attention carry anything a transcript reader lacks?

![Qwen3-32B, 300 prompts: asked right after answering, the self-report's rank correlation with true attention (0.42) is no better than the same model reading the transcript (0.43); added to a reader asked every way at the same time, the self-report adds nothing in any way of asking](../../figures/monitor_logprob.png)

## Answer

No. On the preregistered measure the self-report is no better than the same model reading the transcript, the preregistered pass rule fails (0 of 20 setups pass), and once the reader is asked every wording we asked the self, the self-report adds nothing measurable to it, in any of the 13 ways of asking.

## The story

Qwen3-32B answers a question about four paragraphs. Then we ask where its attention went ("Which paragraph did you attend to most while writing your answer?", a 0–10 rating per paragraph, "which least?", or a free description), and read the probabilities of its answer instead of its sampled answer. We compare that report with the true attention of its answer tokens to each paragraph, and with the same model asked the same question about the conversation given to it as quoted text (the transcript reader).

1. **Preregistered primary truth: the self-report is no better than the reader.** True attention as preregistered (`PREREG.md`, "Truth (primary)") is attention per token of the paragraph (attention mass divided by its token count), from the answer queries, averaged over heads and layers 21–53. Asked right after answering, the per-prompt rank correlation with it is 0.419 for the self-report and 0.432 for the reader (self − reader −0.013 [−0.042, +0.014]). In no way of asking is the self-report above the reader with a CI clear of zero; after true examples it is below.
2. **Preregistered pass rule: FAIL.** The rule (self above both the reader and a transcript classifier, on top-1 accuracy and on rank correlation, Bonferroni over 20 setups) passes in 0 of 20 setups. The classifier on transcript features beats the self-report everywhere (right after, "which most": rank correlation 0.765 vs 0.419; top-1 81% vs 56%). Full tables: `prereg_tables.md` (from `monitor/analyze_monitor.py`).
3. **The self-report adds nothing to a reader asked every way.** We fit, with held-out prompts, a model predicting each paragraph's true-attention rank from the reader's scores for every wording asked at that timing (for example, right after: which most, rating, which least, free), plus paragraph position and length, then add the self-report's scores. The self-report's gain is between −0.006 and +0.013 in all 13 ways of asking, every CI includes zero. Adding a second reader wording to a one-wording reader helps more (typically +0.01 to +0.09) than adding the self-report does (−0.002 to +0.023): the self-report behaves like one more noisy reading of the transcript.
4. **Secondary truth, total attention mass (post hoc).** With total mass (not divided by length) the self-report and reader both correlate better (0.615 and 0.592 right after), and again the self-report adds nothing to the all-wordings reader right after answering (−0.005 to −0.001 for the four wordings, CIs over zero). One setting has a gain with a CI clear of zero: the 0–10 rating asked one turn later, +0.028 [+0.006, +0.049]. That is one of 26 gains (13 ways × 2 truths), picked after seeing them, on the secondary truth, and the same setting under the primary truth gives +0.013 [−0.005, +0.031]; we do not read it as a private signal.
5. **Showing the model examples of its true attention makes its self-report worse.** After 4 or 16 earlier conversations with their measured attention, the reader stays at 0.44–0.45 while the self-report drops (0.40 and 0.38 for "which most?", 0.18 and 0.17 for 0–10 ratings, primary truth). The examples showed length-normalised attention, the same quantity as the primary truth, so the drop is not a mismatch between what the examples taught and what we score. The examples come from other prompts of the same family and shard; a prompt's own label is never in its own prompt, but labels of held-out prompts can appear as examples in training-fold prompts.

## Why the earlier version claimed a "small private signal", and why that was wrong

An earlier version of this page added the self-report to a reader asked only the same wording, scored against total mass, and compared the gain with self-reports shuffled across prompts. The shuffle is the wrong null for a private channel: shuffling each paragraph slot independently breaks the within-prompt structure and every link to the transcript, so any second reading of the transcript, private or not, beats it. The right comparison is a second reader wording, which has no private access by construction. It adds as much as or more than the self-report does, and with all reader wordings in the base the self-report's gain disappears. The shuffle counts are still in `logprob_tables.md`, labelled as a marginal control only, with p = (b + 1)/201.

## Method

- Data: `hypha-prod-mon{0,1,2}/out/hf1.json` and `hf2.json` (feedback reads), 300 prompts in five families of 60 (`PREREG.md` describes the prompts, wordings and reader).
- Answer queries: the query positions that predict each generated answer token, from the last prompt token on (`monitor.py`).
- Truths: "norm" (primary, preregistered) = attention per token of the paragraph; "mass" (secondary) = total attention mass. Both averaged over heads and layers 21–53. With thinking, the final answer's attention (`truth_think.final`). The feedback rows ask right after answering, so their truth is the normal answer's.
- Report score per paragraph: log probability of the label for "which most?" and the free answer (the free description parsed by the same model), minus log probability for "which least?", the expected 0–10 rating for ratings. Reader scores are built the same way.
- (1) Per prompt, Spearman correlation of the four report scores with the four true attention values; mean over prompts with a 10,000-resample bootstrap CI.
- (2) Gain: within each prompt, scores and true attention become ranks, z-scored. A linear model predicts the true-attention rank from the reader's ranks for every wording asked at that timing (after: most, rate, least, free; before, one turn later, after 4 or 16 examples: most, rate; with thinking: most only), paragraph position (one-hot) and log token count; a second model adds the self-report's rank. 10-fold cross-validation grouped by prompt, the split redrawn 10 times and per-prompt results averaged (one split moves the gain by about ±0.01). CI: 2,000 bootstrap resamples of prompts. Alternative null: the gain from adding a different reader wording instead of the self-report to the one-wording reader.
- Code: `monitor/logprob.py` (writes `logprob_results.json`, prints `logprob_tables.md`), `figures/monitor_logprob_fig.py`. Example prompt in the figure: id 173 (rank correlation 0.40 under the primary truth, a typical value).

## Tables

All numbers, both truths: `logprob_tables.md`. Preregistered tables with the pass rule: `prereg_tables.md`.

## Caveats

- Rank correlations over four paragraphs take few distinct values, hence the bands of dots in the figure.
- The base with every reader wording has more features than the self-report adds; the second-reader comparison has the same fit budget as the self-report and is the cleaner control.
