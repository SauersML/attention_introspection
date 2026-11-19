# Label-order audit

Which experiment is behind each figure the README shows and each hero figure, and whether its paragraph labels were in order
([P1]..[P4] in slots 1..4, report options listed in order). "Shuffled" means the labels were assigned to slots at random per prompt
and/or the report options were listed in a random order.

## README figures

| figure | experiment (code) | labels | report options |
|---|---|---|---|
| hero_L28.png | wording ladder, `results/qwen3-32b-dose/dose.py` (`[P{i+1}]` per slot) | in order | no report |
| v_mention_vs_none.png | per-token grid run, `grid/grid_pod.py` | in order | no report |
| dose_full.png | wording ladder, `dose.py` | in order | no report |
| rotate2.png | focus/ignore rotation, `results/qwen3-32b-rotate2/rotate2.py` | in order | no report |
| grid_focus_L28.png | `grid/grid_pod.py` | in order | no report |
| hero_before.png | `results/qwen3-32b-before/before.py` | in order | no report |
| layers.png | `layers3.py` on the ladder's `dose.json` | in order | no report |
| trace3.png | `attn_intro.py` (`results/qwen3-32b/A.json`, `A2.json`) | in order | in order |
| privileged.png | `privileged/privileged.py`, arms A and C, all 5 doses, 100 paragraph prompts (+ the other five kinds at ×10 on its bottom line) | **shuffled** (`c["lab"]` random per prompt); now redrawn in order | **shuffled** (`c["opt"]`); now redrawn in order |
| monitor_logprob.png | `monitor/monitor.py` | in order | **shuffled** (`c["opts"]` random per prompt and wording) |

## Heroes (figures/hero*.png)

| figure | experiment | labels |
|---|---|---|
| hero_L28.png, hero_before.png | as above | in order |
| hero.png, hero_none.png | `attn_intro.py` (`B.json`) | in order |
| hero_told.png, hero_told_top.png, hero_told_yes.png, hero_told_yes_top.png | told injection, `told_inject/told_inject.py` | in order |
| hero_question_top.png | `privileged_inorder/` (arm C at ×5, ×10) | in order |
| hero_question.png | `privileged.py` arm C at ×10 (`results/privileged_shuffled/data`) | **shuffled** |
| hero4*.png | not audited here (left as is) | |

## Other results the README cites

- `results/privileged_words/` (single-word lists, "within ±1% at every dose"): `privileged/privileged_words.py`, labels and options **shuffled**. Not a figure; not rerun.
- `explore/e4` (conflicting sources, not in the README): labels in order (`xlib.py` writes `[P{i+1}]` per slot). What is shuffled is which slot holds the original paragraph and which the altered copy (`order`), a counterbalance of content, not of labels. No rerun needed.
- The sham-label control (`results/qwen3-32b-rotate/sham_label.py`, −0.54 / +0.13 in the README): labels in order.

## Reruns

Shuffled experiments behind README figures, rerun with labels and options in order (`--inorder`, commit d99a33d), everything else identical
(same seed, prompts, doses, arms):

- `privileged.py --inorder`, all 300 prompts, all arms and doses, 6 shards: done, results in `results/privileged_inorder_full/`; the shuffled run moved to `results/privileged_shuffled/`, its figure to `figures/shuffled/privileged.png`. `figures/privileged.png` is now drawn from the in-order run.
- `monitor.py --inorder`, 3 shards, 60 per family as before: done, results in `results/monitor_inorder/` (write-up `RESULT.md`); the shuffled run moved to `results/monitor_shuffled/`, its figure to `figures/shuffled/monitor_logprob.png`. `figures/monitor_logprob.png` is now drawn from the in-order run.

In-order vs shuffled numbers: the README's last section, "Checks: shuffled labels".
