# Is there a privileged channel from attention to self-report? (Qwen3-32B)

Design and pass/fail rules: [PREREG.md](PREREG.md), written and committed (3eed842, then 43f3bc1) before any result existed.
Code: `privileged/privileged.py` (runs on a pod), `privileged/analyze_priv.py` (tables, rules), `figures/privileged_fig.py`.
Data: `data/recs_{0,1,2}.jsonl` (one record per context, every condition), `results.json`, `tables.md` (every table, every setup).
Figure: `figures/privileged.png`.

## Verdict (against the preregistered rules)
**Specific privileged trace, not introspective.** When attention to one part of the context is boosted only while the model writes its answer (answer text identical, report question's own forward pass unmodified), the model's later report of where it attended moves, specifically for the boosted part, with dose, in all five wordings and six of seven kinds of context. The direction is the opposite of introspection: the report becomes less likely to name the boosted part. The effect is small (about −6% at ×5 attention mass, −10% at ×10). Boosting attention while the model reads the report question moves the report strongly toward the boosted part (+31% at ×5, +91% at ×10): the report tracks the attention it is using at report time, not the attention it used while answering.

| rule | result | pass |
|---|---|---|
| R0 manipulation (≥2× mass on the boosted part at ×5) | 3.91× [3.86, 3.96] at the answer queries | yes |
| R1 specific (rotation specificity at ×5, CI excludes 0) | −0.063 [−0.077, −0.050] log-prob (−6%); sign − | yes, sign − |
| R2 dose-dependent, both directions (mean per-context Spearman, sign of R1) | −0.42 [−0.48, −0.36]; the six dose means are in strict order (+0.021, +0.013, 0, −0.020, −0.063, −0.106 for ×¼ … ×10) | yes, monotone |
| R3 wording-robust (≥4 of 5 wordings) | all 5 (W1–W5) | yes |
| R4 general (≥5 of 7 setups) | 6 of 7 (paragraphs-arithmetic, paragraphs-QA, list, code, document, sentences; dialogue's CI touches 0); no setup has the opposite sign | yes |

(R1's CI: `results.json`, key `spec` → `A|W1|all|5.0`: mean −0.0634, lo −0.0767, hi −0.0502.)

## What was done (short; full design in PREREG.md)
- 300 contexts across 7 setups: 4 Wikipedia paragraphs with an unrelated arithmetic question (60) or a question about one paragraph (40); a 4-speaker dialogue (40); a 6-item list (40); 4 Python functions (40); one article in 4 consecutive sections (40); 4 short sentences (40). Labels ([P3], [S2], [I5], [F1]) are randomly assigned to positions per context and options are listed in a random order; results are mapped back to positions.
- The answer is generated once with no intervention and teacher-forced in every condition, so the text is identical across conditions.
- Boost: log(multiplier) added to the pre-softmax attention scores on every key token of one part, all heads, layers 21–53 of 64; multipliers ×¼, ×½, ×2, ×5, ×10; every part boosted in turn (the rotation is the control: a perturbation that disturbs the report without regard to which part was boosted gives 0).
- Arms: A, answer queries only (the prompt's KV cache is shared and asserted unchanged; the bias is asserted off during every report pass, so the only route is the KV cache at the answer positions); B, answer queries and the report question's queries; C, the report question's queries only.
- Report: "Which paragraph (speaker / item / function / section / sentence) did you attend to most / were you focused on / did you rely on most / was most on your mind while writing your answer?", prefilled with the label letter, log-probabilities over the label digits; and a 0–10 rating per part ("how much attention did you pay to paragraph [P3]"), read as the expected rating from digit probabilities.
- Specificity = change in log P(report = boosted part) minus the mean change for the other parts, averaged over the rotation; 95% bootstrap CIs over contexts (10,000 resamples). "%" = exp(specificity) − 1.
- Transcript reader (the same model reading the transcript as quoted text): its shift is 0 by construction, since the text is identical.

## Results

Manipulation check (attention mass on the boosted part, ratio to no boost):

| multiplier | at the answer queries (arms A, B) | at the report question's queries (arms B, C) |
|---|---|---|
| ×¼ | 0.27× [0.27, 0.28] | 0.28× [0.28, 0.28] |
| ×½ | 0.53× [0.53, 0.53] | 0.53× [0.53, 0.53] |
| ×2 | 1.84× [1.83, 1.85] | 1.84× [1.84, 1.85] |
| ×5 | 3.91× [3.86, 3.96] | 3.91× [3.90, 3.93] |
| ×10 | 6.54× [6.42, 6.66] | 6.57× [6.52, 6.62] |

Rotation specificity, primary wording, all 300 contexts (Δ log-prob, with % change in the chance of naming the boosted part):

| multiplier | A: answer only | B: answer + question | C: question only | B − A (paired) |
|---|---|---|---|---|
| ×¼ | +0.02 [+0.01, +0.03] (+2%) | +0.01 [−0.01, +0.03] (+1%) | −0.01 [−0.03, +0.01] (−1%) | −0.01 [−0.03, +0.01] |
| ×½ | +0.01 [+0.01, +0.02] (+1%) | +0.00 [−0.01, +0.02] (+0%) | −0.02 [−0.03, −0.00] (−2%) | −0.01 [−0.03, +0.00] |
| ×2 | −0.02 [−0.03, −0.01] (−2%) | +0.03 [+0.01, +0.05] (+3%) | +0.05 [+0.04, +0.07] (+6%) | +0.05 [+0.03, +0.07] |
| ×5 | −0.06 [−0.08, −0.05] (−6%) | +0.19 [+0.15, +0.24] (+21%) | +0.27 [+0.22, +0.32] (+31%) | +0.25 [+0.21, +0.30] |
| ×10 | −0.11 [−0.12, −0.09] (−10%) | +0.49 [+0.41, +0.57] (+63%) | +0.65 [+0.57, +0.73] (+91%) | +0.60 [+0.52, +0.67] |

Dose (mean per-context Spearman of log multiplier vs specificity): A −0.42 [−0.48, −0.36], B +0.32 [+0.24, +0.41], C +0.48 [+0.41, +0.56].

Arm A at ×5 by wording (W5 in rating points 0–10):

| wording | all | par. arith | par. QA | dialogue | list | code | document | sentences |
|---|---|---|---|---|---|---|---|---|
| attend to most | −0.06 [−0.08, −0.05] | −0.07 [−0.09, −0.05] | −0.19 [−0.25, −0.13] | −0.01 [−0.03, +0.00] | −0.02 [−0.03, −0.00] | −0.08 [−0.10, −0.05] | −0.03 [−0.05, −0.01] | −0.04 [−0.07, −0.00] |
| focused on | −0.07 [−0.08, −0.06] | −0.10 [−0.12, −0.08] | −0.15 [−0.20, −0.10] | −0.02 [−0.04, −0.00] | −0.02 [−0.04, +0.00] | −0.10 [−0.12, −0.08] | −0.05 [−0.08, −0.03] | −0.04 [−0.07, −0.01] |
| relied on | −0.07 [−0.08, −0.06] | −0.08 [−0.10, −0.06] | −0.16 [−0.22, −0.11] | −0.03 [−0.05, −0.01] | −0.04 [−0.06, −0.02] | −0.11 [−0.15, −0.06] | −0.05 [−0.06, −0.03] | −0.04 [−0.07, +0.02] |
| most on your mind | −0.06 [−0.07, −0.05] | −0.07 [−0.09, −0.05] | −0.16 [−0.22, −0.10] | −0.03 [−0.05, −0.01] | −0.01 [−0.03, +0.00] | −0.07 [−0.10, −0.04] | −0.05 [−0.07, −0.04] | −0.04 [−0.06, +0.00] |
| rating 0–10 | −0.01 [−0.02, −0.00] | −0.00 [−0.01, +0.00] | −0.03 [−0.09, +0.00] | −0.03 [−0.06, −0.00] | −0.02 [−0.04, +0.00] | −0.00 [−0.01, −0.00] | +0.00 [−0.01, +0.01] | −0.00 [−0.01, +0.00] |

Arm A by setup and dose, arms B and C by setup, and the 4×4 matrices are in `tables.md`. Arm A's Spearman by setup: paragraphs-arith −0.60, paragraphs-QA −0.77, code −0.50, document −0.36, sentences −0.36, dialogue −0.19, list −0.08 (list's CI includes 0).

## Reading
- There is a privileged route from the attention used while answering to the later report: with the text byte-identical and the report question's own computation untouched, which part was boosted changes the report, specifically (rotation), with dose in both directions (less attention → slightly more likely to be named; more attention → less likely), and in almost every kind of context. A transcript reader cannot see this by construction. Earlier (ROTATE.md) a sham boost on non-paragraph text produced no paragraph-specific shift (−0.04 [−0.27, +0.16] vs −0.54 for the paragraph boost), consistent with the rotation result here.
- That route does not carry an introspective report. Its sign is consistently wrong for "the model reports where it attended", it is small, and it is weakest on the 0–10 rating (about −0.01 points at ×5).
- The report does follow attention, strongly, when the boost happens while the model reads the report question (arm C: +91% at ×10, dose Spearman +0.48). So "which part did you attend to most" is answered from the attention computed for the answer to that question, not read back from the answer's computation. Boosting both (arm B) gives less than question-only at every dose from ×2 up (×10: +0.49 vs +0.65), in the direction of the answer-time trace subtracting.
- The answer-time trace is largest where the answer is long (paragraph QA: −0.19 at ×5; answers there are full sentences, so more answer queries are boosted) and smallest for the dialogue and list setups.

What would overturn "specific privileged trace": a perturbation that singles out a part without changing attention to it (e.g. the boost leaking through the label tokens) could make the same rotation pattern. The boost touches only body tokens (labels and brackets excluded, asserted), and the earlier sham on non-paragraph text gave no specific shift; not checked here: a sham on the label tokens themselves. Null: the rotation control (0 for any non-specific perturbation) and the transcript reader (exactly 0).

## Caveats
- Cached-path check: for 6 contexts (2 per pod), the no-boost report log-probs from the cached path differed from a single uncached forward by up to 0.44 in log-probability (maximum over the labels; low-probability labels are the most sensitive in bf16). Every condition goes through the same cached path, and the none and boosted runs of a context use the same kernels and shapes, so this numerical difference does not enter the comparisons; but it is the same order as the arm A effect, so a reader recomputing without the cache should expect numbers to move slightly.
- Arithmetic answers are 2–3 tokens, so arm A boosts only 3–4 query positions there. The effect is still specific and dose-dependent at that size.
- The boost is a uniform bias over all heads in 33 layers: outside the model's normal operating range. A report that does not read back an unnatural change does not prove the model cannot read back natural variation (results/RESULT.md, experiment A, found no item-level information in natural variation either).
- QA accuracy (no boost) 72.5%; arithmetic 100% in every setup.

## Deviations from the preregistration
- None in the analysis. Before any result existed: the rating question was changed from "[P3]" to "paragraph [P3]" (the first launch was stopped while the model was still downloading); one launch crashed on a bug (the no-boost report pass was not given the segment map; fixed, and smoke-tested on Qwen3-0.6B before relaunch).
- Dropped from the first version of the design (by Scott, before running): the random-residual sham, the third-person and other-model framings, and layer bands. The non-paragraph sham was not rerun (ROTATE.md has it).

## Cost
RunPod, from the ledger: $12.64 for all launches of this experiment, of which $6.73 for the three A100-80GB pods that produced the data (0.98, 1.11, 0.99 h) and $5.91 for stopped, crashed and discarded launches and the 0.6B smoke test. All pods deleted (confirmed in the runner log; no hypha-prod-attnpriv pod running).
