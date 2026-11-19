# Privileged channel from attention to self-report, labels in order (Qwen3-32B)

The privileged-channel experiment (`results/privileged_shuffled/`, plan in its `PREREG.md`) rerun on all 300 contexts with the labels [P1]..[PN] in slots 1..N and the report options listed in order. Everything else is identical: same seed and contexts, all three arms (A: boost at the answer queries only; B: answer and report-question queries; C: report-question queries only), all five multipliers, all five wordings.

Code: `privileged/privileged.py --inorder` via `privileged/run_inorder.sh` (six A100-80GB pods, 50 contexts each; $10.26 on RunPod including stopped launches; all pods deleted). Analysis: `privileged/analyze_priv.py` (the same tables and preregistered rules as the shuffled run) → `results.json`, `tables.md`; `privileged/target_only.py` → `target_only.json`; `privileged/compare_orders.py` → `compare_shuffled.{md,json}` (both runs side by side, paired). Figure: `figures/privileged.png` (`figures/privileged_fig.py`). Data: `data/recs_{0..5}.jsonl`, `data/meta.json`, run logs.

Checks: 300/300 contexts; per-part token counts as in the shuffled run; 293/300 answers identical to the shuffled run (the label numbers in the prompt differ, so the greedy answer can too); arithmetic 100% correct, reading questions 70% (shuffled 72.5%). Cached-path check: max |Δ log p| 0.48 against an uncached forward (shuffled 0.44; same caveat as there).

## Verdict (preregistered rules)
**Specific privileged trace, not introspective**, as in the shuffled run.

| rule | in order | shuffled | pass |
|---|---|---|---|
| R0 manipulation (≥2× mass at ×5) | 3.91× [3.86, 3.95] | 3.91× [3.86, 3.96] | yes |
| R1 specific (arm A, ×5, W1) | −0.070 [−0.082, −0.058] (−6.7%) | −0.063 [−0.077, −0.050] (−6.1%) | yes, sign − |
| R2 dose (mean per-context Spearman) | −0.51 [−0.56, −0.45]; six dose means in strict order | −0.42 [−0.48, −0.36]; strict order | yes |
| R3 wordings (≥4 of 5) | 4 (W1–W4; the 0–10 rating −0.008 [−0.022, +0.002]) | 5 | yes |
| R4 setups (≥5 of 7 at ×5) | 6 (all but dialogue); none opposite | 6 (all but dialogue); none opposite | yes |

## Main numbers (W1; % = exp(rotation specificity) − 1, relative to the other parts)

| multiplier (attention mass) | A: answer only | B: answer + question | C: question only |
|---|---|---|---|
| ×¼ (0.28×) | +3.1% [+2.2, +3.9] | +3% [+0, +5] | +0.8% [−1.4, +2.9] |
| ×½ (0.53×) | +1.9% [+1.3, +2.6] | +1% [−1, +2] | −1.1% [−2.5, +0.3] |
| ×2 (1.84×) | −3.0% [−3.7, −2.3] | +3% [+1, +5] | +6.5% [+4.5, +8.5] |
| ×5 (3.9×) | −6.7% [−7.9, −5.6] | +26% [+20, +33] | +40.6% [+33.7, +47.9] |
| ×10 (6.5×) | −11.1% [−12.5, −9.6] | +89% [+75, +105] | +138.7% [+119.3, +160.2] |

The boosted part's own probability (not relative): arm A −3.5% at ×5, −5.4% [−7.1, −3.8] at ×10; arm C +33% and +104% [+90, +120].
Arm A at ×10 by kind: paragraphs-QA −23%, document −12%, paragraphs-arithmetic −14%, code −11%, sentences −8%, list −4%, dialogue −3%; every interval excludes 0 (shuffled: lists −1%, interval includes 0). Arm C at ×10 by kind: from +53% (paragraphs-QA) to +533% (document). B and C by kind and dose, every wording, and the 4×4 matrices: `tables.md`.

## Against the shuffled run (paired, same prompts; `compare_shuffled.md`)
- Arm A, the main result: no difference at any dose on the 100 paragraph prompts (every paired interval includes 0) or on all 300 at ×10 (−1.1% [−2.5, +0.2]). Label order does not produce or change the answer-time trace. One kind differs: document sections at ×10, −12.0% against −6.2% (paired −6.3% [−9.5, −2.9]).
- Arm C is larger in order: all 300 at ×10, +139% against +91% (in order 25% [16, 34] higher, prompt by prompt); paragraphs-QA +53% [+27, +89] against +11.5% [−6.5, +35]; document +533% against +269%. At low multipliers on paragraphs arm C is slightly positive in order (+7.8% at ×¼) where it was not distinguishable from 0 shuffled; by kind, arm C at ×¼ and ×½ has large opposite-signed values (document −0.21, paragraphs-arithmetic +0.16 in log-prob at ×¼), so at low doses the reading-time effect is not monotone in dose within a kind.
- The reading of the experiment is unchanged: attention used while answering leaves a small, specific trace in the later report with the wrong sign for introspection; attention while reading the report question moves the report strongly toward the boosted part, more so with labels in order.

What would overturn this: a difference between runs in arm A larger than the paired intervals allow (checked: none on paragraphs, all kinds −1.1% [−2.5, +0.2]). Null: the rotation design gives 0 for any perturbation that does not depend on which part is boosted; the transcript reader is exactly 0 since the text is identical.
