# Qwen/Qwen3-32B (layers 64, band 21–53, B1=1.0, B2=2.0)

### A: QA about one paragraph (n=200; truth = argmax per-token attention, att_first, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 92.5% [88.5, 96.0] |
| transcript-only reader (same model) | 92.5% [89.0, 96.0] |
| transcript-feature classifier (5-fold CV) | 92.5% [88.5, 96.0] |
| transcript features + reader (5-fold CV) | 92.5% [88.5, 96.0] |
| transcript features + reader + self-report (5-fold CV) | 92.5% [88.5, 96.0] |
| task-relevant paragraph (by construction) | 92.5% [88.5, 96.0] |
| majority position (P4, in-sample) | 29.5% [23.5, 36.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 0.0% [0.0, 0.0] |

Where self and reader disagree (n=0): self right 0, reader right 0. Truth distribution P1–P4 [49, 43, 49, 59]; self [48, 46, 52, 54]; reader [48, 46, 52, 54].

### A: summarize all four (n=200; truth = argmax per-token attention, att_first, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 48.0% [41.0, 55.0] |
| transcript-only reader (same model) | 67.0% [60.5, 73.5] |
| transcript-feature classifier (5-fold CV) | 93.0% [89.0, 96.5] |
| transcript features + reader (5-fold CV) | 93.0% [89.0, 96.5] |
| transcript features + reader + self-report (5-fold CV) | 93.0% [89.5, 96.5] |
| majority position (P1, in-sample) | 93.0% [89.5, 96.5] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -19.0% [-25.5, -12.5] |

Where self and reader disagree (n=67): self right 8, reader right 46. Truth distribution P1–P4 [186, 6, 2, 6]; self [102, 46, 24, 28]; reader [141, 35, 13, 11].

### A: unrelated arithmetic (paragraphs irrelevant) (n=200; truth = argmax per-token attention, att_first, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 34.0% [27.5, 40.5] |
| transcript-only reader (same model) | 30.5% [24.0, 37.0] |
| transcript-feature classifier (5-fold CV) | 39.5% [33.0, 46.0] |
| transcript features + reader (5-fold CV) | 35.5% [29.0, 42.0] |
| transcript features + reader + self-report (5-fold CV) | 39.5% [33.0, 46.0] |
| majority position (P4, in-sample) | 38.0% [31.5, 45.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 3.5% [-4.5, 11.5] |

Where self and reader disagree (n=106): self right 38, reader right 31. Truth distribution P1–P4 [64, 26, 34, 76]; self [3, 34, 22, 141]; reader [63, 43, 39, 55].

### A: all three tasks (n=600; truth = argmax per-token attention, att_first, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 58.2% [54.2, 62.0] |
| transcript-only reader (same model) | 63.3% [59.3, 67.2] |
| transcript-feature classifier (5-fold CV) | 71.0% [67.5, 74.7] |
| transcript features + reader (5-fold CV) | 73.3% [69.8, 76.8] |
| transcript features + reader + self-report (5-fold CV) | 73.8% [70.3, 77.3] |
| task-relevant paragraph (by construction) | 47.7% [43.8, 51.7] |
| majority position (P1, in-sample) | 49.8% [45.8, 53.8] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | -5.2% [-8.7, -1.7] |

Where self and reader disagree (n=173): self right 46, reader right 77. Truth distribution P1–P4 [299, 75, 85, 141]; self [153, 126, 98, 223]; reader [252, 124, 104, 120].

### A: QA about one paragraph (n=200; truth = argmax per-token attention, att_mean, layers 21–53)

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

### A: summarize all four (n=200; truth = argmax per-token attention, att_mean, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 18.0% [13.0, 23.5] |
| transcript-only reader (same model) | 18.0% [13.0, 23.5] |
| transcript-feature classifier (5-fold CV) | 63.5% [57.0, 70.0] |
| transcript features + reader (5-fold CV) | 61.5% [54.5, 68.0] |
| transcript features + reader + self-report (5-fold CV) | 63.5% [56.5, 70.0] |
| majority position (P4, in-sample) | 59.0% [52.0, 66.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 0.0% [-5.0, 5.0] |

Where self and reader disagree (n=67): self right 13, reader right 13. Truth distribution P1–P4 [24, 23, 35, 118]; self [102, 46, 24, 28]; reader [141, 35, 13, 11].

### A: unrelated arithmetic (paragraphs irrelevant) (n=200; truth = argmax per-token attention, att_mean, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 54.0% [47.0, 61.0] |
| transcript-only reader (same model) | 26.0% [20.0, 32.0] |
| transcript-feature classifier (5-fold CV) | 69.5% [63.0, 75.5] |
| transcript features + reader (5-fold CV) | 68.5% [62.0, 75.0] |
| transcript features + reader + self-report (5-fold CV) | 69.5% [63.0, 76.0] |
| majority position (P4, in-sample) | 69.5% [63.0, 76.0] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 28.0% [20.5, 36.0] |

Where self and reader disagree (n=106): self right 67, reader right 11. Truth distribution P1–P4 [19, 17, 25, 139]; self [3, 34, 22, 141]; reader [63, 43, 39, 55].

### A: all three tasks (n=600; truth = argmax per-token attention, att_mean, layers 21–53)

| predictor | agreement with most-attended paragraph [95% CI] |
|---|---|
| self-report | 57.3% [53.3, 61.2] |
| transcript-only reader (same model) | 48.0% [44.0, 52.2] |
| transcript-feature classifier (5-fold CV) | 75.8% [72.3, 79.2] |
| transcript features + reader (5-fold CV) | 75.3% [71.8, 78.7] |
| transcript features + reader + self-report (5-fold CV) | 76.0% [72.5, 79.3] |
| task-relevant paragraph (by construction) | 50.2% [46.2, 54.3] |
| majority position (P4, in-sample) | 51.8% [47.8, 55.8] |
| chance | 25.0% [25.0, 25.0] |
| self minus reader (paired) | 9.3% [6.2, 12.5] |

Where self and reader disagree (n=173): self right 80, reader right 24. Truth distribution P1–P4 [91, 86, 112, 311]; self [153, 126, 98, 223]; reader [252, 124, 104, 120].

### A2 (arith): boost b=1.0 on a random non-target paragraph Pk (n=200; layers 21–53, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 2.384 [2.370, 2.399] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 2.319 [2.302, 2.337] |
| Pk is most-attended (first query): base → boost | 25.5% [19.5, 31.5] → 88.5% [84.0, 92.5] |
| self-report picks Pk: base → boost | 25.5% [19.5, 31.5] → 25.0% [19.0, 31.0] |
| transcript reader picks Pk: base → boost | 18.5% [13.0, 24.0] → 18.5% [13.5, 24.0] |
| Δ self − Δ reader (difference-in-differences, picks) | -0.5% [-3.5, 2.5] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.017 [-0.028, -0.007] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.266 [-0.323, -0.210] |
| same, transcript reader | 0.000 [0.000, 0.000] |
| self − reader (log-prob shift) | -0.266 [-0.324, -0.210] |
| answer text identical to unboosted | 100.0% [100.0, 100.0] (200/200) |
| identical-answer subset: Δ self picks Pk | -0.5% [-3.5, 2.5] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.017 [-0.027, -0.007] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.266 [-0.324, -0.209] |
| text leak: Δ answer word-overlap with Pk | 0.000 [0.000, 0.000] |
| QA correct: base → boost | 99.5% [98.5, 100.0] → 99.5% [98.5, 100.0] |

### A2 (arith): boost b=2.0 on a random non-target paragraph Pk (n=200; layers 21–53, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 5.222 [5.155, 5.287] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 4.894 [4.817, 4.972] |
| Pk is most-attended (first query): base → boost | 25.5% [19.5, 31.5] → 100.0% [100.0, 100.0] |
| self-report picks Pk: base → boost | 25.5% [19.5, 31.5] → 23.0% [17.0, 29.0] |
| transcript reader picks Pk: base → boost | 18.5% [13.0, 24.0] → 18.5% [13.5, 24.0] |
| Δ self − Δ reader (difference-in-differences, picks) | -2.5% [-6.0, 1.0] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.022 [-0.042, -0.002] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.360 [-0.486, -0.237] |
| same, transcript reader | -0.003 [-0.008, 0.000] |
| self − reader (log-prob shift) | -0.358 [-0.486, -0.232] |
| answer text identical to unboosted | 99.5% [98.5, 100.0] (199/200) |
| identical-answer subset: Δ self picks Pk | -2.5% [-6.0, 1.0] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.022 [-0.042, -0.002] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.361 [-0.488, -0.235] |
| text leak: Δ answer word-overlap with Pk | 0.000 [0.000, 0.000] |
| QA correct: base → boost | 99.5% [98.5, 100.0] → 100.0% [100.0, 100.0] |

### A2 (qa): boost b=1.0 on a random non-target paragraph Pk (n=200; layers 21–53, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 2.403 [2.390, 2.415] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 2.217 [2.201, 2.233] |
| Pk is most-attended (first query): base → boost | 1.5% [0.0, 3.5] → 38.5% [32.0, 45.5] |
| self-report picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| transcript reader picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| Δ self − Δ reader (difference-in-differences, picks) | 0.0% [0.0, 0.0] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.000 [-0.000, -0.000] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.213 [-0.252, -0.177] |
| same, transcript reader | -0.002 [-0.021, 0.014] |
| self − reader (log-prob shift) | -0.211 [-0.249, -0.173] |
| answer text identical to unboosted | 87.5% [83.0, 92.0] (175/200) |
| identical-answer subset: Δ self picks Pk | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.000 [-0.000, -0.000] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.219 [-0.257, -0.182] |
| text leak: Δ answer word-overlap with Pk | -0.000 [-0.001, 0.000] |
| QA correct: base → boost | 83.0% [78.0, 88.0] → 82.5% [77.0, 87.5] |

### A2 (qa): boost b=2.0 on a random non-target paragraph Pk (n=200; layers 21–53, all heads, task+answer queries)

| quantity | value [95% CI] |
|---|---|
| attention mass ratio on Pk, boost/base (first answer query) | 5.271 [5.214, 5.328] |
| attention mass ratio on Pk, boost/base (mean over answer queries) | 4.433 [4.370, 4.495] |
| Pk is most-attended (first query): base → boost | 1.5% [0.0, 3.5] → 99.5% [98.5, 100.0] |
| self-report picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| transcript reader picks Pk: base → boost | 0.0% [0.0, 0.0] → 0.0% [0.0, 0.0] |
| Δ self − Δ reader (difference-in-differences, picks) | 0.0% [0.0, 0.0] |
| Δ P(Pk) self − Δ P(Pk) reader (probabilities) | -0.000 [-0.000, -0.000] |
| Pk-specific shift in log P(report), self (Δlog P(Pk) − mean Δlog P(other non-target)) | -0.450 [-0.526, -0.376] |
| same, transcript reader | -0.009 [-0.048, 0.026] |
| self − reader (log-prob shift) | -0.440 [-0.523, -0.363] |
| answer text identical to unboosted | 71.5% [65.0, 77.5] (143/200) |
| identical-answer subset: Δ self picks Pk | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ reader picks Pk (text identical, so 0 expected) | 0.0% [0.0, 0.0] |
| identical-answer subset: Δ self P(Pk) | -0.000 [-0.000, -0.000] |
| identical-answer subset: Pk-specific log-prob shift, self (reader = 0 by construction) | -0.458 [-0.548, -0.374] |
| text leak: Δ answer word-overlap with Pk | -0.001 [-0.003, 0.001] |
| QA correct: base → boost | 83.0% [77.5, 88.0] → 83.0% [77.5, 88.0] |

### B (qa): instructed attention to Pk (k ≠ question paragraph), n=200 contexts paired with no instruction

| condition | mass ratio on Pk vs none, first answer query [95% CI] | mass ratio on Pk vs none, mean over answer queries | mass ratio on question paragraph Pj (mean queries) | answer identical to none | correct |
|---|---|---|---|---|---|
| none | 1 | 1 | 1 | 100% | 83.0% [77.5, 88.0] |
| focus | 0.926 [0.918, 0.934] | 1.105 [1.088, 1.123] | 0.924 [0.901, 0.945] | 49.0% [42.0, 56.0] | 84.5% [79.5, 89.5] |
| ignore | 0.897 [0.890, 0.903] | 0.977 [0.968, 0.985] | 0.993 [0.984, 1.003] | 62.0% [55.0, 68.5] | 85.0% [80.0, 90.0] |
| mention | 0.836 [0.830, 0.842] | 1.002 [0.993, 1.011] | 0.972 [0.965, 0.979] | 63.5% [57.0, 70.0] | 84.5% [79.5, 89.5] |

focus vs ignore (att_first, paired): mass ratio 1.033 [1.025, 1.041]

ignore vs mention (att_first, paired): mass ratio 1.072 [1.067, 1.078]

focus vs mention (att_first, paired): mass ratio 1.108 [1.100, 1.116]

focus vs ignore (att_mean, paired): mass ratio 1.132 [1.116, 1.148]

ignore vs mention (att_mean, paired): mass ratio 0.975 [0.969, 0.981]

focus vs mention (att_mean, paired): mass ratio 1.103 [1.089, 1.118]

Per-layer mean log mass ratio on Pk vs none (mean over answer queries), layer: focus / ignore / mention
L0:-0.05/-0.06/-0.05 L1:-0.07/-0.07/-0.03 L2:-0.07/-0.07/-0.03 L3:-0.09/-0.09/-0.05 L4:-0.10/-0.10/-0.05 L5:-0.05/-0.05/-0.02 L6:-0.07/-0.07/-0.04 L7:-0.07/-0.06/-0.07 L8:-0.08/-0.07/-0.09 L9:-0.08/-0.07/-0.08 L10:-0.10/-0.09/-0.09 L11:-0.10/-0.08/-0.10 L12:-0.06/-0.05/-0.05 L13:-0.05/-0.05/-0.05 L14:-0.05/-0.05/-0.05 L15:-0.04/-0.03/-0.05 L16:-0.04/-0.03/-0.06 L17:-0.05/-0.04/-0.06 L18:-0.02/-0.01/-0.03 L19:-0.06/-0.03/-0.06 L20:-0.07/-0.05/-0.07 L21:+0.05/+0.02/+0.05 L22:+0.04/+0.02/+0.03 L23:-0.02/+0.00/-0.01 L24:+0.16/+0.02/+0.05 L25:+0.07/-0.00/+0.01 L26:+0.28/+0.00/+0.07 L27:+0.15/+0.01/+0.04 L28:+0.25/-0.02/+0.06 L29:+0.07/-0.04/+0.00 L30:+0.02/-0.03/-0.02 L31:+0.02/-0.04/-0.03 L32:+0.00/-0.05/-0.04 L33:+0.04/-0.01/+0.00 L34:-0.03/-0.04/-0.04 L35:+0.03/-0.01/-0.01 L36:+0.16/+0.01/+0.05 L37:+0.13/+0.01/+0.03 L38:+0.05/-0.04/-0.01 L39:+0.05/-0.02/-0.00 L40:+0.08/-0.02/-0.02 L41:+0.11/-0.01/-0.00 L42:+0.18/-0.01/+0.01 L43:+0.08/-0.04/-0.03 L44:+0.02/-0.03/-0.05 L45:+0.29/-0.14/-0.03 L46:+0.16/-0.11/-0.05 L47:+0.13/-0.00/-0.05 L48:+0.12/-0.13/-0.07 L49:+0.03/-0.13/-0.08 L50:+0.12/-0.14/-0.07 L51:+0.04/-0.05/-0.05 L52:+0.18/-0.03/-0.00 L53:+0.12/-0.07/-0.01 L54:+0.08/-0.04/-0.01 L55:+0.08/-0.01/-0.00 L56:+0.10/+0.00/+0.02 L57:+0.10/-0.01/-0.01 L58:+0.20/-0.04/+0.00 L59:+0.24/+0.01/+0.03 L60:+0.12/-0.00/-0.01 L61:+0.15/+0.03/+0.02 L62:+0.08/+0.03/-0.01 L63:+0.01/+0.04/-0.03

### B (arith): instructed attention to Pk (k ≠ question paragraph), n=200 contexts paired with no instruction

| condition | mass ratio on Pk vs none, first answer query [95% CI] | mass ratio on Pk vs none, mean over answer queries | mass ratio on question paragraph Pj (mean queries) | answer identical to none | correct |
|---|---|---|---|---|---|
| none | 1 | 1 | 1 | 100% | 99.5% [98.5, 100.0] |
| focus | 1.032 [1.016, 1.049] | 1.389 [1.354, 1.426] | 0.993 [0.972, 1.014] | 99.0% [97.5, 100.0] | 99.0% [97.5, 100.0] |
| ignore | 0.883 [0.877, 0.889] | 0.994 [0.978, 1.010] | 0.957 [0.945, 0.970] | 100.0% [100.0, 100.0] | 99.5% [98.5, 100.0] |
| mention | 0.930 [0.922, 0.939] | 1.100 [1.080, 1.120] | 0.961 [0.947, 0.976] | 100.0% [100.0, 100.0] | 99.5% [98.5, 100.0] |

focus vs ignore (att_first, paired): mass ratio 1.169 [1.152, 1.186]

ignore vs mention (att_first, paired): mass ratio 0.949 [0.942, 0.956]

focus vs mention (att_first, paired): mass ratio 1.109 [1.097, 1.122]

focus vs ignore (att_mean, paired): mass ratio 1.398 [1.377, 1.418]

ignore vs mention (att_mean, paired): mass ratio 0.904 [0.898, 0.910]

focus vs mention (att_mean, paired): mass ratio 1.263 [1.246, 1.281]

Per-layer mean log mass ratio on Pk vs none (mean over answer queries), layer: focus / ignore / mention
L0:-0.08/-0.09/-0.09 L1:-0.08/-0.08/-0.05 L2:-0.07/-0.07/-0.02 L3:-0.08/-0.07/-0.04 L4:-0.07/-0.07/-0.03 L5:-0.05/-0.05/-0.02 L6:-0.07/-0.07/-0.05 L7:-0.08/-0.08/-0.08 L8:-0.08/-0.07/-0.08 L9:-0.08/-0.09/-0.09 L10:-0.13/-0.13/-0.10 L11:-0.06/-0.06/-0.06 L12:-0.03/-0.01/-0.02 L13:-0.04/-0.02/+0.00 L14:-0.06/-0.05/-0.04 L15:-0.02/-0.02/-0.02 L16:-0.04/-0.04/-0.06 L17:-0.04/-0.03/-0.07 L18:-0.05/-0.03/-0.07 L19:-0.10/-0.07/-0.12 L20:-0.10/-0.07/-0.12 L21:+0.16/+0.01/+0.10 L22:+0.07/-0.02/+0.01 L23:+0.08/+0.02/+0.03 L24:+0.75/+0.17/+0.34 L25:+0.34/+0.08/+0.07 L26:+0.78/+0.05/+0.15 L27:+0.58/+0.03/+0.16 L28:+0.95/+0.01/+0.18 L29:+0.50/+0.05/+0.15 L30:+0.18/-0.04/+0.00 L31:+0.34/+0.04/+0.07 L32:+0.33/-0.02/+0.06 L33:+0.26/+0.04/+0.09 L34:+0.09/-0.05/-0.02 L35:+0.22/-0.02/+0.04 L36:+0.63/+0.10/+0.27 L37:+0.34/+0.01/+0.08 L38:+0.18/-0.05/+0.00 L39:+0.11/-0.06/+0.03 L40:+0.12/-0.10/-0.02 L41:+0.38/-0.03/+0.12 L42:+0.21/-0.09/+0.06 L43:+0.07/-0.16/+0.01 L44:+0.04/-0.12/-0.06 L45:+0.26/-0.05/+0.12 L46:+0.08/-0.11/-0.07 L47:+0.14/-0.07/+0.05 L48:+0.29/+0.02/+0.14 L49:+0.39/+0.05/+0.12 L50:+0.59/-0.02/+0.24 L51:+0.28/+0.02/+0.14 L52:+0.44/+0.03/+0.35 L53:+0.26/+0.04/+0.26 L54:+0.21/+0.02/+0.14 L55:+0.17/+0.02/+0.08 L56:+0.12/-0.01/+0.06 L57:+0.27/+0.04/+0.15 L58:+0.45/+0.09/+0.29 L59:+0.44/+0.09/+0.25 L60:+0.21/+0.03/+0.10 L61:+0.26/+0.06/+0.14 L62:+0.25/+0.06/+0.15 L63:+0.02/-0.03/-0.00
