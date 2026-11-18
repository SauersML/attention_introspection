# Instruction once before the paragraphs vs after them (Qwen3-32B, 150 dose contexts, seed 7, layers 21–53, answer-token queries, teacher-forced on the no-instruction answer)

After-position versions rerun on the same pod; max relative difference from the dose run: none 0.0e+00, a_focus 0.0e+00, a_ignore 0.0e+00, a_mention 0.0e+00

## focus
- before_vs_before_mention: +16.3% [+14.3, +18.3], 97% of contexts > 0, sign-flip p 5e-06
- before_vs_none: +23.2% [+20.1, +26.0], 96% of contexts > 0, sign-flip p 5e-06
- after_vs_after_mention: +47.3% [+44.5, +50.1], 100% of contexts > 0, sign-flip p 5e-06
- after_vs_none: +61.5% [+56.9, +66.1], 100% of contexts > 0, sign-flip p 5e-06
- before_vs_after_same_instruction: -21.5% [-24.3, -18.7], 14% of contexts > 0, sign-flip p 5e-06
- diff_of_effects_vs_own_mention_pts: -31.0% [-34.5, -27.5], 4% of contexts > 0, sign-flip p 5e-06
## ignore
- before_vs_before_mention: -6.1% [-8.1, -2.9], 5% of contexts > 0, sign-flip p 5e-06
- before_vs_none: -0.7% [-3.3, +2.9], 45% of contexts > 0, sign-flip p 0.8
- after_vs_after_mention: -14.3% [-15.1, -13.4], 0% of contexts > 0, sign-flip p 5e-06
- after_vs_none: -6.4% [-8.1, -4.8], 33% of contexts > 0, sign-flip p 5e-06
- before_vs_after_same_instruction: +7.2% [+4.0, +10.8], 67% of contexts > 0, sign-flip p 1e-05
- diff_of_effects_vs_own_mention_pts: +8.2% [+6.1, +11.3], 92% of contexts > 0, sign-flip p 5e-06

- before mention vs none: +6.0% [+4.1, +7.7], 91% of contexts > 0, sign-flip p 5e-06
- after mention vs none: +9.4% [+7.4, +11.4], 76% of contexts > 0, sign-flip p 5e-06
- free answers identical to no-instruction: b_mention 100.0%, b_focus 100.0%, b_ignore 98.7%