# Told injection as a literal two-turn conversation (Qwen3-32B, 240 prompts)

Same intro, paragraphs, seed and boost as results/told_inject (keys of paragraph k, layers 21-53, while reading and answering). Only the question changes:
- Turn 1: "Trial 1: Do you notice your attention being boosted onto one of the paragraphs? Answer with just Yes or No."
- Turn 2, only after a Yes: "Which paragraph? Answer with just the label, e.g. [P1]." (boost still on)

Attention on the boosted paragraph at m=20: x10.2 (told_inject: x9.9), so the boost replicates.

| | no injection | m=10 | m=20 |
|---|---|---|---|
| greedy "Yes" | 1 / 240 | 2 / 960 | 17 / 960 |
| P(Yes), renormalised over Yes/No | 16.5% | 19.6% | 22.7% |
| turn 2 names the boosted paragraph | | 1 of 2 | 6 of 17 (chance 25%) |

Reading: with a bare Yes/No answer, Qwen nearly always says "No." even under the boost; the probability of "Yes" still rises (about as much as in told_inject), but the greedy reply rarely crosses. Too few Yes replies for the turn-2 column. The single-turn version (told_inject), whose answer format lists "Yes, paragraph [Pk]" as options, gives 13% -> 22% greedy Yes. The Yes rate depends strongly on the answer format.
