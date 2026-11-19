# Salience, honest prompt, meaning addition size-matched (Qwen3-32B, 240 contexts)

Same honest prompt as results/salience_honest (Qwen is told we either boost attention onto one paragraph or add that paragraph's
content into its activations). Arms: none, A (attention to the paragraph boosted x20, layers 21-53), B (paragraph meaning added at
the old scale), Bs (same vector times 0.8). Mean residual change per layer, Bs / A = 0.99 (B / A = 1.24), so Bs is the size-matched
meaning arm and is the one used in figures/salience.png.

Means over 240 contexts (boosted arms averaged over target paragraph P1..P4), 95% bootstrap CI (10,000 resamples),
p = two-sided sign-flip over contexts (20,000 flips; 5e-05 is the floor).

| measure | none | attention (A) | meaning, size-matched (Bs) | meaning, old scale (B) |
|---|---|---|---|---|
| reply starts "Yes" (says something changed) | 44.6 [38.3, 50.8] | 54.7 [51.7, 57.8] | 37.6 [34.9, 40.3] | 35.4 [32.7, 38.1] |
| top paragraph digit = boosted paragraph (chance 25) | | 54.4 [51.9, 56.9] | 35.6 [33.5, 37.8] | 37.2 [35.0, 39.5] |
| P("attention") vs "activations" after "Yes, paragraph [Pk]," (chance 50) | | 73.1 [72.1, 74.0] | 63.6 [62.6, 64.6] | 66.7 [65.7, 67.8] |

Paired differences:
- says something changed: A - Bs +17.1 [13.6, 20.5], p < 1e-4; A - none +10.1 [4.3, 15.9], p = 0.0008; Bs - none -7.0 [-12.4, -1.5], p = 0.016
- right paragraph: A - Bs +18.8 [15.7, 21.8], p < 1e-4
- says "attention": A - Bs +9.5 [8.5, 10.4], p < 1e-4 (was +6.4 with the larger B)

Reading: with the meaning addition shrunk to exactly the attention boost's activation change, the attention-vs-meaning gap on
"attention" holds and grows (6.4 -> 9.5 points), so the earlier gap was not from the meaning push being bigger.
