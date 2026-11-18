# Qwen/Qwen3-8B (layers 36, band 12–30, B1=1.0, B2=2.0)

### A: QA about one paragraph (n=200; truth = argmax per-token attention, att_first, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 95.5% [92.5, 98.0] |
| transcript-only reader (same model) | 95.5% [92.5, 98.0] |
| transcript-feature classifier (5-fold CV) | 95.0% [91.5, 98.0] |
| transcript features + reader (5-fold CV) | 95.0% [92.0, 98.0] |
| transcript features + reader + self-report (5-fold CV) | 95.5% [92.5, 98.0] |
| task-relevant paragraph (by construction) | 95.5% [92.5, 98.0] |
| majority position (P4, in-sample) | 27.5% [21.5, 34.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 0.0% [0.0, 0.0] |

Where self and reader disagree (n=0): self right 0, reader right 0. Truth distribution P1–P4 [51, 46, 48, 55]; self [48, 46, 52, 54]; reader [48, 46, 52, 54].

### A: summarize all four (n=200; truth = argmax per-token attention, att_first, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 50.0% [43.0, 57.0] |
| transcript-only reader (same model) | 78.0% [72.0, 83.5] |
| transcript-feature classifier (5-fold CV) | 96.5% [93.5, 99.0] |
| transcript features + reader (5-fold CV) | 96.5% [94.0, 99.0] |
| transcript features + reader + self-report (5-fold CV) | 96.5% [93.5, 99.0] |
| majority position (P1, in-sample) | 96.5% [93.5, 99.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -28.0% [-34.5, -21.5] |

Where self and reader disagree (n=67): self right 2, reader right 58. Truth distribution P1–P4 [193, 2, 0, 5]; self [103, 43, 51, 3]; reader [163, 14, 17, 6].

### A: unrelated arithmetic (paragraphs irrelevant) (n=200; truth = argmax per-token attention, att_first, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 49.0% [42.0, 56.0] |
| transcript-only reader (same model) | 51.5% [44.5, 58.5] |
| transcript-feature classifier (5-fold CV) | 55.0% [48.0, 62.0] |
| transcript features + reader (5-fold CV) | 55.0% [48.0, 62.0] |
| transcript features + reader + self-report (5-fold CV) | 53.5% [46.5, 60.5] |
| majority position (P4, in-sample) | 49.0% [42.0, 56.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -2.5% [-7.0, 2.0] |

Where self and reader disagree (n=29): self right 8, reader right 13. Truth distribution P1–P4 [84, 7, 11, 98]; self [7, 10, 16, 167]; reader [6, 2, 4, 188].

### A: all three tasks (n=600; truth = argmax per-token attention, att_first, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 64.8% [61.0, 68.7] |
| transcript-only reader (same model) | 75.0% [71.5, 78.5] |
| transcript-feature classifier (5-fold CV) | 75.5% [72.2, 79.0] |
| transcript features + reader (5-fold CV) | 80.3% [77.2, 83.5] |
| transcript features + reader + self-report (5-fold CV) | 81.3% [78.2, 84.3] |
| task-relevant paragraph (by construction) | 48.2% [44.2, 52.2] |
| majority position (P1, in-sample) | 54.7% [50.7, 58.7] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -10.2% [-13.0, -7.5] |

Where self and reader disagree (n=96): self right 10, reader right 71. Truth distribution P1–P4 [328, 55, 59, 158]; self [158, 99, 119, 224]; reader [217, 62, 73, 248].

### A: QA about one paragraph (n=200; truth = argmax per-token attention, att_mean, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 100.0% [100.0, 100.0] |
| transcript-only reader (same model) | 100.0% [100.0, 100.0] |
| transcript-feature classifier (5-fold CV) | 100.0% [100.0, 100.0] |
| transcript features + reader (5-fold CV) | 100.0% [100.0, 100.0] |
| transcript features + reader + self-report (5-fold CV) | 100.0% [100.0, 100.0] |
| task-relevant paragraph (by construction) | 100.0% [100.0, 100.0] |
| majority position (P4, in-sample) | 27.0% [21.0, 33.5] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 0.0% [0.0, 0.0] |

Where self and reader disagree (n=0): self right 0, reader right 0. Truth distribution P1–P4 [48, 46, 52, 54]; self [48, 46, 52, 54]; reader [48, 46, 52, 54].

### A: summarize all four (n=200; truth = argmax per-token attention, att_mean, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 27.5% [21.5, 33.5] |
| transcript-only reader (same model) | 34.5% [28.0, 41.0] |
| transcript-feature classifier (5-fold CV) | 59.0% [52.0, 66.0] |
| transcript features + reader (5-fold CV) | 59.5% [52.5, 66.0] |
| transcript features + reader + self-report (5-fold CV) | 57.5% [50.5, 64.0] |
| majority position (P1, in-sample) | 39.5% [32.5, 46.5] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -7.0% [-13.0, -1.5] |

Where self and reader disagree (n=67): self right 10, reader right 24. Truth distribution P1–P4 [79, 25, 26, 70]; self [103, 43, 51, 3]; reader [163, 14, 17, 6].

### A: unrelated arithmetic (paragraphs irrelevant) (n=200; truth = argmax per-token attention, att_mean, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 64.5% [58.0, 71.0] |
| transcript-only reader (same model) | 68.0% [61.5, 74.5] |
| transcript-feature classifier (5-fold CV) | 65.0% [58.5, 71.5] |
| transcript features + reader (5-fold CV) | 68.0% [61.5, 74.0] |
| transcript features + reader + self-report (5-fold CV) | 68.0% [61.5, 74.5] |
| majority position (P4, in-sample) | 65.0% [58.0, 71.5] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -3.5% [-8.5, 1.5] |

Where self and reader disagree (n=29): self right 9, reader right 16. Truth distribution P1–P4 [38, 11, 21, 130]; self [7, 10, 16, 167]; reader [6, 2, 4, 188].

### A: all three tasks (n=600; truth = argmax per-token attention, att_mean, layers 12–30)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 64.0% [60.2, 67.8] |
| transcript-only reader (same model) | 67.5% [63.7, 71.3] |
| transcript-feature classifier (5-fold CV) | 73.3% [69.7, 76.8] |
| transcript features + reader (5-fold CV) | 73.3% [69.8, 76.8] |
| transcript features + reader + self-report (5-fold CV) | 73.8% [70.3, 77.3] |
| task-relevant paragraph (by construction) | 52.2% [48.2, 56.3] |
| majority position (P4, in-sample) | 42.3% [38.5, 46.3] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -3.5% [-6.0, -1.0] |

Where self and reader disagree (n=96): self right 19, reader right 40. Truth distribution P1–P4 [165, 82, 99, 254]; self [158, 99, 119, 224]; reader [217, 62, 73, 248].

### A2 (arith): boost b=1.0 on a random non-target paragraph Pk (n=200; layers 12–30, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 2.333 [2.312, 2.356] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 2.337 [2.314, 2.359] |
| Pk is most-attended (first query): base → boost | 29.0% [23.0, 35.5] → 84.0% [79.0, 89.0] |
| self-report picks Pk: base → boost | 26.0% [20.0, 32.0] → 26.5% [20.5, 33.0] |
| transcript reader picks Pk: base → boost | 27.5% [21.5, 33.5] → 27.5% [21.5, 34.0] |
| Δ self − Δ reader (difference-in-differences, picks) | 0.5% [-2.5, 3.5] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.010 [-0.026, 0.005] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.280 [-0.397, -0.166] |
| same, transcript reader | 0.000 [0.000, 0.000] |
| self − reader (log-prob shift) | -0.280 [-0.396, -0.165] |
| answer text identical to unboosted | 100.0% [100.0, 100.0] (200/200) |
| identical-answer subset: Δ self picks Pk | 0.5% [-2.5, 3.5] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.010 [-0.026, 0.005] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.280 [-0.395, -0.167] |
| text leak: Δ answer word-overlap with Pk | 0.000 [0.000, 0.000] |
| QA correct: base → boost | 100.0% [100.0, 100.0] → 100.0% [100.0, 100.0] |

### A2 (arith): boost b=2.0 on a random non-target paragraph Pk (n=200; layers 12–30, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 5.067 [4.974, 5.161] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 5.015 [4.915, 5.120] |
| Pk is most-attended (first query): base → boost | 29.0% [23.0, 35.5] → 94.0% [90.5, 97.0] |
| self-report picks Pk: base → boost | 26.0% [20.0, 32.0] → 26.0% [20.0, 32.5] |
| transcript reader picks Pk: base → boost | 27.5% [21.5, 34.0] → 27.5% [21.5, 33.5] |
| Δ self − Δ reader (difference-in-differences, picks) | 0.0% [-3.5, 3.5] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.003 [-0.031, 0.024] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.435 [-0.711, -0.166] |
| same, transcript reader | 0.000 [0.000, 0.000] |
| self − reader (log-prob shift) | -0.435 [-0.709, -0.171] |
| answer text identical to unboosted | 100.0% [100.0, 100.0] (200/200) |
| identical-answer subset: Δ self picks Pk | 0.0% [-3.5, 3.5] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.003 [-0.030, 0.024] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.435 [-0.711, -0.174] |
| text leak: Δ answer word-overlap with Pk | 0.000 [0.000, 0.000] |
| QA correct: base → boost | 100.0% [100.0, 100.0] → 100.0% [100.0, 100.0] |

### A2 (qa): boost b=1.0 on a random non-target paragraph Pk (n=200; layers 12–30, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 2.298 [2.275, 2.321] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 2.231 [2.206, 2.254] |
| Pk is most-attended (first query): base → boost | 1.5% [0.0, 3.5] → 7.5% [4.0, 11.5] |
| self-report picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| transcript reader picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| Δ self − Δ reader (difference-in-differences, picks) | 0.0% [0.0, 0.0] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.000 [-0.000, 0.000] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.260 [-0.310, -0.212] |
| same, transcript reader | -0.005 [-0.024, 0.013] |
| self − reader (log-prob shift) | -0.255 [-0.305, -0.208] |
| answer text identical to unboosted | 87.5% [83.0, 92.0] (175/200) |
| identical-answer subset: Δ self picks Pk | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.000 [-0.000, 0.000] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.260 [-0.306, -0.214] |
| text leak: Δ answer word-overlap with Pk | -0.001 [-0.002, 0.000] |
| QA correct: base → boost | 81.5% [76.0, 87.0] → 81.0% [75.5, 86.0] |

### A2 (qa): boost b=2.0 on a random non-target paragraph Pk (n=200; layers 12–30, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 4.878 [4.786, 4.973] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 4.547 [4.465, 4.632] |
| Pk is most-attended (first query): base → boost | 1.5% [0.0, 3.5] → 39.0% [32.5, 46.0] |
| self-report picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| transcript reader picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| Δ self − Δ reader (difference-in-differences, picks) | 0.0% [0.0, 0.0] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.000 [-0.000, 0.000] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.596 [-0.697, -0.501] |
| same, transcript reader | -0.003 [-0.031, 0.027] |
| self − reader (log-prob shift) | -0.593 [-0.692, -0.499] |
| answer text identical to unboosted | 73.5% [67.5, 79.5] (147/200) |
| identical-answer subset: Δ self picks Pk | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.000 [-0.000, 0.000] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.568 [-0.679, -0.466] |
| text leak: Δ answer word-overlap with Pk | -0.001 [-0.005, 0.001] |
| QA correct: base → boost | 81.5% [76.0, 86.5] → 82.0% [76.5, 87.0] |

### B (qa): instructed attention to Pk (k ≠ question paragraph), n=200 contexts paired with no instruction

| condition | mass ratio on Pk vs none, first answer query [95% CI] | mass ratio on Pk vs none, mean over answer queries | mass ratio on question paragraph Pj (mean queries) | answer identical to none | correct |
|---|---|---|---|---|---|
| none | 1 | 1 | 1 | 100% | 81.5% [76.0, 86.5] |
| focus | 0.987 [0.968, 1.008] | 1.081 [1.052, 1.117] | 0.881 [0.831, 0.927] | 55.0% [48.0, 62.0] | 77.5% [71.5, 83.0] |
| ignore | 0.853 [0.844, 0.861] | 0.979 [0.967, 0.991] | 1.023 [0.999, 1.048] | 65.0% [58.5, 71.5] | 82.5% [77.0, 87.5] |
| mention | 0.864 [0.854, 0.874] | 0.986 [0.973, 0.998] | 0.986 [0.967, 1.004] | 67.0% [60.5, 73.5] | 82.5% [77.0, 87.5] |

focus vs ignore (att_first, paired): mass ratio 1.158 [1.135, 1.182]

ignore vs mention (att_first, paired): mass ratio 0.987 [0.980, 0.994]

focus vs mention (att_first, paired): mass ratio 1.143 [1.122, 1.164]

focus vs ignore (att_mean, paired): mass ratio 1.104 [1.075, 1.141]

ignore vs mention (att_mean, paired): mass ratio 0.993 [0.984, 1.004]

focus vs mention (att_mean, paired): mass ratio 1.097 [1.068, 1.134]

Per-layer mean log mass ratio on Pk vs none (mean over answer queries), layer: focus / ignore / mention
L0:-0.05/-0.08/-0.06 L1:-0.06/-0.04/-0.05 L2:-0.03/-0.02/-0.03 L3:-0.03/-0.02/-0.03 L4:-0.04/-0.03/-0.03 L5:-0.03/-0.02/-0.03 L6:-0.05/-0.02/-0.03 L7:-0.05/-0.02/-0.03 L8:-0.02/+0.03/+0.00 L9:+0.01/+0.05/+0.05 L10:-0.03/-0.01/-0.01 L11:-0.05/-0.03/-0.06 L12:-0.02/-0.02/-0.05 L13:+0.06/-0.01/+0.00 L14:+0.04/-0.02/-0.02 L15:+0.05/-0.02/-0.01 L16:+0.04/-0.00/-0.01 L17:+0.09/-0.03/-0.01 L18:+0.17/-0.07/-0.03 L19:+0.11/-0.05/-0.05 L20:+0.15/-0.02/-0.03 L21:+0.11/-0.06/-0.04 L22:+0.06/-0.01/-0.01 L23:+0.15/-0.02/-0.01 L24:+0.16/-0.03/-0.01 L25:+0.05/-0.02/-0.02 L26:+0.10/-0.02/+0.00 L27:+0.03/-0.01/+0.00 L28:+0.08/-0.05/-0.02 L29:+0.11/-0.03/+0.00 L30:+0.08/-0.06/-0.04 L31:+0.11/-0.04/-0.01 L32:+0.07/-0.04/-0.03 L33:+0.10/-0.03/-0.01 L34:+0.05/-0.04/-0.03 L35:-0.09/-0.07/-0.09

### B (arith): instructed attention to Pk (k ≠ question paragraph), n=200 contexts paired with no instruction

| condition | mass ratio on Pk vs none, first answer query [95% CI] | mass ratio on Pk vs none, mean over answer queries | mass ratio on question paragraph Pj (mean queries) | answer identical to none | correct |
|---|---|---|---|---|---|
| none | 1 | 1 | 1 | 100% | 100.0% [100.0, 100.0] |
| focus | 1.166 [1.128, 1.205] | 1.032 [1.009, 1.052] | 0.893 [0.874, 0.909] | 97.5% [95.0, 99.5] | 97.5% [95.0, 99.5] |
| ignore | 0.834 [0.818, 0.846] | 0.910 [0.891, 0.924] | 0.917 [0.908, 0.925] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] |
| mention | 0.841 [0.824, 0.855] | 0.928 [0.909, 0.944] | 0.862 [0.845, 0.876] | 99.0% [97.5, 100.0] | 99.0% [97.5, 100.0] |

focus vs ignore (att_first, paired): mass ratio 1.398 [1.363, 1.436]

ignore vs mention (att_first, paired): mass ratio 0.992 [0.985, 0.998]

focus vs mention (att_first, paired): mass ratio 1.386 [1.355, 1.419]

focus vs ignore (att_mean, paired): mass ratio 1.135 [1.125, 1.144]

ignore vs mention (att_mean, paired): mass ratio 0.980 [0.973, 0.987]

focus vs mention (att_mean, paired): mass ratio 1.112 [1.106, 1.119]

Per-layer mean log mass ratio on Pk vs none (mean over answer queries), layer: focus / ignore / mention
L0:-0.07/-0.08/-0.06 L1:-0.05/-0.05/-0.06 L2:-0.03/-0.02/-0.04 L3:-0.03/-0.03/-0.04 L4:-0.04/-0.03/-0.05 L5:-0.02/-0.02/-0.03 L6:-0.02/-0.02/-0.03 L7:-0.03/-0.03/-0.04 L8:-0.00/-0.02/-0.06 L9:-0.04/-0.03/-0.04 L10:-0.06/-0.06/-0.03 L11:-0.14/-0.11/-0.13 L12:-0.11/-0.13/-0.15 L13:+0.16/-0.05/+0.00 L14:+0.07/-0.06/-0.07 L15:+0.16/-0.03/-0.02 L16:+0.03/-0.06/-0.10 L17:+0.16/-0.11/-0.01 L18:+0.10/-0.14/-0.14 L19:-0.00/-0.17/-0.16 L20:-0.03/-0.17/-0.17 L21:-0.09/-0.15/-0.12 L22:-0.07/-0.08/-0.13 L23:-0.11/-0.13/-0.13 L24:+0.01/-0.09/+0.01 L25:-0.03/-0.10/-0.09 L26:-0.01/-0.09/-0.06 L27:-0.02/-0.09/-0.05 L28:-0.03/-0.11/-0.08 L29:+0.08/-0.09/-0.02 L30:+0.01/-0.11/-0.09 L31:+0.02/-0.11/-0.07 L32:-0.03/-0.14/-0.11 L33:+0.03/-0.11/-0.06 L34:+0.05/-0.09/-0.04 L35:-0.12/-0.13/-0.11
