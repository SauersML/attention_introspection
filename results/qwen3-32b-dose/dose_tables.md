n=150 contexts, band layers 21-53, mean over answer-token queries, teacher-forced on the no-instruction answer

| condition | vs mention | share > 0 | vs none | share > 0 | vs mention, first query | free answer differs from none |
|---|---|---|---|---|---|---|
| 0_none | -7.3% [-9.1, -5.6] | 24% | — | — | +8.0% [+6.9, +9.1] | 0.0% |
| 1_mention | — | — | +9.4% [+7.4, +11.4] | 76% | — | 0.0% |
| 2_glance | +38.0% [+35.5, +40.4] | 100% | +50.9% [+47.1, +54.7] | 99% | +6.9% [+6.1, +7.7] | 1.3% |
| 3_focus | +47.3% [+44.5, +50.1] | 100% | +61.5% [+56.9, +66.1] | 100% | +6.4% [+5.5, +7.2] | 0.7% |
| 4_focus_strongly | +38.0% [+35.6, +40.5] | 100% | +51.3% [+47.1, +55.3] | 99% | +3.6% [+2.6, +4.6] | 1.3% |
| 5_very_strongly_only | +20.9% [+18.9, +23.0] | 100% | +32.5% [+29.0, +35.9] | 95% | -6.2% [-7.4, -4.9] | 4.7% |
| 6_twice | +202.0% [+189.9, +214.3] | 100% | +233.7% [+216.6, +251.2] | 100% | +45.6% [+40.9, +50.2] | 5.3% |
| 7_little_less | +1.0% [+0.1, +1.9] | 53% | +10.5% [+8.3, +12.7] | 76% | -5.9% [-6.6, -5.1] | 0.0% |
| 8_do_not | -0.8% [-1.6, +0.1] | 40% | +8.5% [+6.4, +10.6] | 74% | -2.9% [-3.6, -2.2] | 0.7% |
| 9_completely_ignore | -14.3% [-15.1, -13.4] | 0% | -6.4% [-8.1, -4.8] | 33% | -10.3% [-11.1, -9.5] | 0.7% |

up-ladder: means vs mention 2_glance +38.0%, 3_focus +47.3%, 4_focus_strongly +38.0%, 5_very_strongly_only +20.9%, 6_twice +202.0%
Spearman(order, mean) = +0.30, bootstrap 95% CI [+0.10, +0.30], per-context Spearman mean 
step 2_glance->3_focus: +6.8% [+6.0, +7.6], share > 0 93%
step 3_focus->4_focus_strongly: -6.1% [-6.7, -5.4], share > 0 5%
step 4_focus_strongly->5_very_strongly_only: -12.2% [-13.0, -11.4], share > 0 1%
step 5_very_strongly_only->6_twice: +149.9% [+140.4, +159.6], share > 0 100%

down-ladder: means vs mention 7_little_less +1.0%, 8_do_not -0.8%, 9_completely_ignore -14.3%
Spearman(order, mean) = -1.00, bootstrap 95% CI [-1.00, -1.00], per-context Spearman mean 
step 7_little_less->8_do_not: -1.7% [-2.2, -1.3], share > 0 21%
step 8_do_not->9_completely_ignore: -13.5% [-14.2, -12.9], share > 0 1%
step 1_mention->7_little_less: +1.0% [+0.1, +1.9]

per layer, % change vs mention (mean over contexts):
| layer | 0_none | 2_glance | 3_focus | 4_focus_strongly | 5_very_strongly_only | 6_twice | 7_little_less | 8_do_not | 9_completely_ignore |
|---|---|---|---|---|---|---|---|---|---|
| 0 | +8 | +4 | +3 | +4 | -1 | -0 | +2 | +2 | -0 |
| 4 | +3 | -0 | -1 | -0 | -1 | -4 | -0 | -0 | -0 |
| 8 | +9 | +3 | +1 | +1 | +4 | +1 | +1 | +2 | +6 |
| 12 | +3 | +2 | +0 | -1 | +0 | -5 | +1 | +0 | +4 |
| 16 | +7 | +7 | +4 | +4 | +3 | +7 | +4 | +2 | +5 |
| 20 | +13 | +4 | -0 | +3 | -3 | +16 | +3 | +4 | +0 |
| 24 | -26 | +85 | +98 | +78 | +46 | +288 | +16 | +8 | -17 |
| 28 | -12 | +184 | +242 | +192 | +112 | +1156 | +10 | +5 | -25 |
| 32 | -2 | +46 | +58 | +44 | +30 | +150 | +2 | -2 | -12 |
| 36 | -21 | +63 | +83 | +60 | +41 | +305 | -1 | -5 | -23 |
| 40 | +4 | +33 | +32 | +30 | +20 | +287 | +7 | +2 | -8 |
| 44 | +8 | +19 | +17 | +15 | +3 | +135 | +5 | +2 | -8 |
| 48 | -10 | +27 | +26 | +17 | +13 | +84 | -8 | -1 | -13 |
| 52 | -28 | +14 | +13 | +6 | +3 | +81 | -22 | -21 | -29 |
| 56 | -5 | +3 | +6 | +3 | -5 | +103 | -9 | -6 | -16 |
| 60 | -9 | +1 | +5 | +3 | -5 | +86 | -10 | -8 | -16 |

forced answer correct 100.0%; free answers correct: 0_none 100.0%, 1_mention 100.0%, 2_glance 98.7%, 3_focus 99.3%, 4_focus_strongly 98.7%, 5_very_strongly_only 95.3%, 6_twice 94.7%, 7_little_less 100.0%, 8_do_not 99.3%, 9_completely_ignore 99.3%