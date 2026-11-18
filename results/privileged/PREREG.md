# Preregistration: is there a privileged channel from attention to self-report? (Qwen3-32B)

Written 18 Nov 2026 before any run of `privileged.py`. Version 2 (design changes from Scott, made before anything ran: the rotation is the main control; random-residual sham, third-person/other-model framings and layer bands dropped; report-question steering arms and diverse setups added). Nothing below is changed after results arrive; any later change is listed under "Deviations" in RESULT.md.

## Background
Experiment A2 (results/RESULT.md) added a bias to the pre-softmax attention scores on one paragraph Pk during the task and answer, held the answer text identical, and asked "Which paragraph were you paying the most attention to?". The report's log-probability for Pk moved by −0.27 / −0.36 (arithmetic; Pk-specific, relative to the other paragraphs) while a transcript reader (same model, fresh pass over the identical text) moved by 0. A sham bias of the same size on non-paragraph text (header and question, 100 contexts, ROTATE.md in compute/attn_intro) gave a Pk-specific shift of −0.04 [−0.27, +0.16] against −0.54 [−0.72, −0.36] for the paragraph boost; that sham is not rerun here.

This experiment asks whether the trace is a privileged channel: specific to the boosted segment (rotation), dose-dependent in both directions, robust to wording, general across kinds of context, and how it compares with steering attention while the model is answering the report question itself.

## Model
Qwen/Qwen3-32B, bf16, eager attention recomputed in a patched attention function (as in attn_intro.py). Chat template, thinking off.

## Setups (300 contexts, seed 11; segments come from SQuAD-train text unless stated)
Every context is a list of N labelled segments followed by a task. Labels are a letter plus a number ([P1], [S2], ...). In every context the label numbers are assigned to segment positions by a random permutation (so label and position are decorrelated) and the options in each report question are listed in a second random order. All results are mapped back to segment positions.
1. paragraphs-arith (60): 4 paragraphs [P1..P4] (60–110 tokens, 4 different articles); task "Unrelated question: what is X + Y? Answer with just the number."
2. paragraphs-QA (40): the same layout; task: a SQuAD question about one paragraph Pj.
3. dialogue (40): a conversation of 4 speakers [S1..S4], each speaker has two turns (first and second sentence of a paragraph from that speaker's own article), turn order shuffled per round; a speaker's segment is the union of their two turns. Task: arithmetic.
4. list (40): a list of 6 items [I1..I6], each a SQuAD question from a different article. Task: arithmetic.
5. code (40): 4 Python functions [F1..F4] drawn from a fixed pool of hand-written functions (label in a comment line above each function). Task: arithmetic.
6. document (40): one article split into 4 consecutive sections [S1..S4] (4 consecutive paragraphs of the same SQuAD article, 40–110 tokens each). Task: arithmetic.
7. sentences (40): 4 short sentences [S1..S4] (first sentences of paragraphs from 4 articles, 8–40 tokens). Task: arithmetic.
A segment's tokens are its body only (label and brackets excluded).

The answer is generated once, greedily, with no intervention, and then teacher-forced (identical tokens) in every condition, so the transcript text is byte-identical across conditions of a context.

## Intervention
Boost of segment k by multiplier m: add log(m) to the pre-softmax attention scores of every key token in segment k, all 64 heads, layers 21–53 (the band used before). Doses m ∈ {0.25, 0.5, 2, 5, 10} plus m = 1 (none). Every segment is boosted in turn at every dose (the rotation).

Three arms, which differ only in which queries receive the bias:
- A, answer only (the privileged-channel test): the answer queries (the last prompt token through the last answer token). The report question's forward pass is unmodified. The prompt's KV cache (positions before the answer queries) is computed once with no intervention and shared by every condition; the code asserts it is unchanged after each condition, asserts that the bias is off whenever a report question is run in this arm, and asserts that every biased query lies inside the answer span. So the only route from the intervention to the report is the KV cache at the answer positions.
- B, answer + report question: arm A's answer-query bias, plus the same bias for every query of the report question turn (including the final prefilled position that produces the label).
- C, report question only: the answer KV cache is the unboosted one; the bias applies only to the report question turn's queries.
Check of the cached path: on the first 2 contexts of each shard, the cached no-intervention report log-probs are compared with a single uncached forward over the same tokens; the maximum difference is logged.

## Report questions (a new user turn after the answer; X = the setup's segment noun and letter)
- Choice (prefill the label letter, read the probabilities of the label digits, renormalised over the N labels, as log-probabilities):
  W1 "Which {noun} did you attend to most while writing your answer?" (primary)
  W2 "Which {noun} were you focused on while writing your answer?"
  W3 "Which {noun} did you rely on most while writing your answer?"
  W4 "Which {noun} was most on your mind while writing your answer?"
  each followed by "Answer with just one of {options in shuffled order}."
- Rating W5: for each label, "On a scale from 0 to 10, how much attention did you pay to [Xl] while writing your answer? Answer with just the number."; expected rating from the digit probabilities ("10" read as "1" followed by "0").
- Transcript reader: fresh pass, the same model given the user message and the answer as quoted text, asked W1 about "the assistant". The text is identical across conditions, so its shift is exactly 0 by construction; reported as the zero line.

## Measures
- Δ_c(m) = log P(report = segment m | condition c) − log P(report = m | none), per context.
- Specificity of a condition boosting segment k: Δ(k) − mean over the other segments of Δ(m). The rotation is the control: a perturbation that disturbs the report without reference to which segment was boosted gives 0 on average over the rotation.
- Per context and dose: the rotation specificity = mean over k of that quantity (diagonal minus mean off-diagonal of the N×N matrix). For W5, the same on expected ratings (points 0–10).
- Manipulation check: attention mass on the boosted segment, head-averaged, layers 21–53, at the queries the arm biases (answer queries for A; report-question queries for C), ratio boost / none, geometric mean over contexts and segments.
- Dose: per context, Spearman correlation between log m (six doses; none = 0) and rotation specificity; mean over contexts. Also the mean specificity at each dose.
- 95% bootstrap CIs over contexts (10,000 resamples). Primary analysis pools all 300 contexts; each setup is also reported separately.

## Pass/fail rules (arm A, primary wording W1, pooled 300 contexts unless stated)
- R0 manipulation: the m = 5 boost raises attention mass on the boosted segment at the answer queries by at least 2×. If not, uninformative.
- R1 specific: rotation specificity at m = 5, CI excludes 0. Its sign defines the direction: + = the report moves toward the boosted segment (what introspection predicts), − = away.
- R2 dose-dependent, both directions: the mean per-context Spearman has a CI excluding 0, with sign matching R1 (a + direction means larger m gives more positive specificity, so m < 1 should give the opposite sign of m > 1). "Monotone" is additionally reported as whether the six dose means are in strict order.
- R3 wording-robust: at least 4 of the 5 wordings (W1–W5) have rotation specificity at m = 5 with a CI excluding 0 in R1's direction.
- R4 general: at least 5 of the 7 setups have rotation specificity at m = 5 with a CI excluding 0 in R1's direction.
- Arms B and C: descriptive, no rule. Reported: their specificity at m = 5, sign, dose curve, and the difference B − A (what steering during the question adds).

## Verdicts (arm A)
- "Privileged introspective channel": R0, R1 with sign +, R2, R3 and R4 pass.
- "Specific privileged trace, not introspective": R0, R1 with sign −, and R2 pass (R3 and R4 say how robust and general).
- "Specific trace, but not shown to be dose-dependent": R1 passes, R2 fails.
- "No specific channel": R1 fails.
If setups disagree in sign at m = 5 (CIs excluding 0 on both sides), the verdict says so.
