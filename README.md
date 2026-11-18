# attention_introspection

Can a language model see and steer its own attention? Experiments on Qwen3-32B (and Qwen3-8B).


## Setup
Each prompt has four short Wikipedia paragraphs labelled [P1]..[P4] and an unrelated arithmetic question. Attention is read from the model directly: how much of the answer tokens' attention lands on each paragraph (averaged over heads and a middle band of layers).

## What we find
- **Steering:** being rerun with every instruction naming "paragraph [Pk]" (the first runs said a bare "[Pk]", which reads as a label, so their numbers and figures were removed). The rotation run already uses "Focus your attention on paragraph [Pk].": the named paragraph gets the biggest boost in 97% of questions (figures/rotate.png).
- **Reporting:** asked which paragraph it attended to most, the model does no better than the same model reading the transcript.
- **A trace through the cache** (no instruction involved, so unaffected): boosting attention on one paragraph (with the answer text identical) moves the model's report about its attention, in log-probability, consistently and with dose. The sign is negative: it becomes less likely to name the boosted paragraph. A sham boost on non-paragraph text gives no such shift (figures/trace3.png).


## Files
- attn_intro.py: the experiment (patched eager attention, the boost intervention, the report).
- analyze.py: tables and intervals.
- results/: per-item json and tables for each model; results/RESULT.md is the full write-up with all numbers and caveats.
- figures/: figures and the scripts that make them.
