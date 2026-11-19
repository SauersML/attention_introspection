# Attention monitoring in Qwen3-32B: preregistration

Written 18 Nov 2026, before any data of this experiment was collected. Code: `monitor/monitor.py` (pod), `monitor/analyze_monitor.py` (tables, verdict, figure). Builds on experiment A (results/RESULT.md): Qwen3-32B's report of which paragraph it attended to most was no better than the same model reading the transcript. Here we try hard to elicit a real self-report: lopsided natural attention, six report wordings, three timings, thinking on/off, and in-context feedback with the model's true measured attention.

## Question
Can Qwen3-32B report where its own attention went (which paragraph its answer tokens attended to most, and the ranking/distribution over paragraphs) better than (a) the same model reading the identical transcript and (b) a classifier on transcript text?

## Model and measurement
- Qwen/Qwen3-32B, bf16, greedy decoding. Generation in vLLM (0.19.1), then every attention measurement and every report probability read in transformers (5.18) with a patched attention function that computes the exact softmax attention weights for the recorded query rows.
- Context: four paragraphs labelled [P1]..[P4], then a task. Paragraph tokens = body tokens, label excluded.
- Truth (primary): for each layer, attention weights averaged over the 64 heads, summed over a paragraph's tokens and divided by its token count (length-normalised), averaged over the answer queries (the query positions that predict each generated answer token, from the last prompt token on) and over layers 21–53 (the band used in experiment A). The four values define the truth ranking; the most-attended paragraph is its argmax and the least-attended its argmin.
- Truth (secondary, reported, not used for the pass rule): total attention mass (not divided by length), layers 21–53; and both measures over all 64 layers.
- With thinking on, the answer queries are all generated tokens (thinking and final answer); final-answer-only truth is also reported.

## Contexts: 300, five families of 60 (each designed to make attention lopsided without the text saying where)
1. Distractor QA: four paragraphs from the SAME Wikipedia article (SQuAD), a SQuAD question about one of them.
2. Multi-hop QA: four SQuAD paragraphs from different articles; four inserted fictional bridge sentences: "[Person A] was born in [Town X]." in one paragraph, "[Town X] is known for [fact]." in another, and the same pair for a second person and town in the other two. Question: "What is the town where [Person A] was born known for?"
3. Arithmetic over paragraph numbers: four SQuAD paragraphs that each contain a number in digits; "Take the first number written in digits in paragraph [Pa] and add the first number written in digits in paragraph [Pb]."
4. Summarise one: "Summarize the paragraph about [article title] in one sentence."
5. Long vs short: four paragraphs of very different lengths (about 25–50, 60–100, 130–190 and 230–340 tokens), random order; "Write a short title (at most eight words) that covers all four paragraphs."

## Report wordings (six)
Every self question begins: "By attention I mean your transformer attention weights: how much the tokens of your answer attended to the tokens of each paragraph." The reader question begins with the same sentence about "the assistant". Labels in every option list are in a shuffled order per context and wording.
1. most: "Which paragraph did you attend to most …? Answer with just one of: P3, P1, P4, P2." Assistant turn prefilled with "P"; score = probability of each label digit (renormalised over the four).
2. least: "Which paragraph did you rely on least …?" Same read; scored against the truth's least-attended paragraph; for ranking, score = minus the probability.
3. rank: "Rank the paragraphs by how much attention you paid … from most to least." Generated; labels parsed in order.
4. pct: "Estimate the percentage of your attention that went to each paragraph …" Generated; "Pk: x%" parsed.
5. rate: "On a scale from 0 to 10, how much attention did you pay to paragraph [Pk] …?" asked separately for each paragraph; score = expected value read from the digit probabilities (with "10" resolved by a second read after "1").
6. free: "In a few sentences, describe how your attention was distributed across the four paragraphs …" Generated; then parsed by the same model given only the description (not the paragraphs): "According to this description, which paragraph received the most attention?" (label probabilities).
Parse failure (no valid label, or no percentages) counts as wrong for top-1 and 0 for rank correlation, for self and reader alike.

## Timings and other factors (setups)
| setup | timing | thinking | feedback examples k | wordings |
|---|---|---|---|---|
| after | new user turn right after the answer | off | 0 | all six |
| before | the question is appended to the task message; the model predicts before answering; truth = the attention of its normal answer to the same message | off | 0 | most, rate, pct |
| later | after the answer, one unrelated user question and the model's own reply, then the question | off | 0 | most, rate, pct |
| think | thinking on for the answer and for the report (report: thinking up to 300 tokens, then the answer is read) | on | 0 | most, pct |
| feedback k=4, k=16 | after; the report turn first shows k earlier conversations (same family, other contexts) with the model's true measured attention per paragraph (length-normalised band truth, in percent, and the most-attended label), labelled as measured from inside the model | off | 4, 16 | most, rate, pct |
20 setups in total (6 + 3 + 3 + 2 + 3 + 3), each on all 300 contexts.

## Baselines
- (a) Transcript reader: the same model, given the identical transcript as quoted text ("Below is a conversation between a user and an AI assistant (Qwen3-32B) …"), asked the same wording about the assistant, read the same way, with the same thinking mode and the same k feedback examples. In "before", the reader sees only the user message. In "later", it sees the later turn too. With thinking on, the transcript includes the assistant's reasoning text.
- (b) Transcript classifier: per paragraph features (position, log token count, word overlap of the task with the paragraph, word overlap of the answer and of the reasoning text with the paragraph, count of numbers, whether the task names its label); fitted per family with 5-fold cross-validation over contexts. Top-1: conditional logit on the most-attended (or least-attended for "least") paragraph. Ranking: ridge regression of the log share of truth; score = prediction. In "before", answer and reasoning features are dropped.

## Metrics
- Top-1 accuracy: self's argmax vs truth's argmax (argmin for "least").
- Rank correlation: Spearman between the four scores and the four truth values, per context, averaged over contexts.
- Paired differences self − reader and self − classifier, per context; 95% bootstrap CI (10,000 resamples of contexts). Per family numbers are reported descriptively.

## Pass rule (the one claim)
A setup passes if, on all 300 contexts, self − reader AND self − classifier are both above 0 for top-1 accuracy AND for rank correlation, each with its paired bootstrap CI clear of 0 after Bonferroni correction over the 20 setups (two-sided 99.75% CI lower bound > 0). The experiment PASSES (the model shows a privileged self-report of its attention) if at least one setup passes; otherwise it FAILS. Uncorrected 95% CIs are reported beside.

The one observation that would overturn a pass: the self advantage vanishes under the secondary truths (total mass, all layers) or is carried by one family; both are reported. A fail would be overturned if the reader or classifier had access the self lacks; by construction they see only text the self also had.

## Amendment before any data (18 Nov, 01:00)
Every prompt that names a paragraph says "paragraph [Pk]", never a bare "[Pk]" (arithmetic task, rate question, and the feedback records: "Measured attention: paragraph [P1] 12%, … (most attended: paragraph [P2])"). Answer-format option lists ("Answer with just one of: P3, P1, P4, P2") are unchanged. The smoke test (10 contexts) ran with the old wording and is used only to check the code.
