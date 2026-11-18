# Per-token attention grid and instruction violins (Qwen3-32B, arithmetic task)

100 new contexts (seed 11; construction as in rotate.py: four SQuAD paragraphs [P1]..[P4], then "Unrelated question: what is X + Y? Answer with just the number."), 13 versions each:
none; and for k = 1..4: "Note that one of the paragraphs is labelled [Pk]." / "Focus your attention on paragraph [Pk]." / "Do not pay attention to paragraph [Pk]."
Measure: eager attention, head mean, answer-token queries, teacher-forced on the no-instruction answer, layers 21-53 of 64; per-token values saved for every token of all four paragraphs.

- `grid_pod.py`, `grid_run.sh`: the GPU run (pod hypha-prod-attngridv2a, one A100-80GB). `data/G.json`: per-context records (per-token and per-paragraph band means for all 13 versions, free answers).
- `grid.py`: figure `figures/grid.png` (4 contexts, picked just below/above the pool mean effect of focus and of ignore vs mention); `data/grid_results.json` (picks, checks), `data/grid_summary.json` (4x4 named x measured matrices, geometric mean, 95% bootstrap; per-context band means), `data/violin_data.json` (per context and k).
- `vfigs.py`: `figures/v_none.png`, `figures/v_mention.png`, `figures/v_mention_vs_none.png`; `data/violins.json` (means, CIs, sign-flip p, per-pair values).

Checks: paragraph means from per-token data equal the span means (max relative difference 2.5e-7); 1193 of 1200 free (greedy) answers under an instruction equal the no-instruction answer; no-instruction answer correct in 99 of 100.
