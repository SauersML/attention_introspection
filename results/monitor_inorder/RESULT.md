# Attention monitoring with report options in order (Qwen3-32B)

The attention-monitoring experiment (plan `results/monitor_shuffled/PREREG.md`, first run and write-up `results/monitor_shuffled/LOGPROB.md`) rerun on the same 300 prompts with the report options listed in order (P1, P2, P3, P4). The paragraph labels were already [P1]..[P4] in slot order in the first run; only the order of the options in the report question was random per prompt and wording. Everything else is identical: same prompts, seed, wordings, timings, feedback examples and reader.

Code: `monitor/monitor.py --inorder` via `monitor/run_inorder.sh` (3 shards, 60 prompts per family). Data: `data/shard{0,1,2}/` (`contexts.json`, `gen1.json`, `gen2.json`, `hf1.json`, `hf2.json`). Analysis: `monitor/analyze_monitor.py` → `prereg_results.json`, `prereg_tables.md`; `monitor/logprob.py` → `logprob_results.json`, `logprob_tables.md` (full stdout in `logprob_stdout.txt`); `monitor/compare_orders.py` → `compare_shuffled.{md,json}` (both runs side by side, paired by prompt). Figure: `figures/monitor_logprob.png` (`figures/monitor_logprob_fig.py`); the shuffled run's figure is `figures/shuffled/monitor_logprob.png`.

Checks: 300/300 prompts; 283/300 answers identical to the shuffled run. The answer prompt is the same in both runs (options only appear in the later report question), so the 17 differences are generation nondeterminism; they are rewordings of the same answer (a bold mark moved, "Replicated CDs" for "Compact discs").

## Answer

Same conclusion as the shuffled run, slightly stronger: the self-report does not track the model's attention better than the same model reading the transcript, and adds nothing to a reader asked every way. Two numbers the README cited change:

- Right after answering, "which paragraph most?", primary truth: self-report 0.42, reader 0.49 (shuffled: 0.42 and 0.43). The self-report is now below the reader with a CI clear of zero (−0.064 [−0.096, −0.032]). The self-report is unchanged prompt by prompt (+0.003 [−0.030, +0.034]); the reader went up (+0.053 [+0.020, +0.087], p = 0.002).
- The post-hoc survivor no longer survives. Under total attention mass, the 0–10 rating asked one turn later gave +0.028 [+0.006, +0.049] in the shuffled run; in order it is +0.004 [−0.014, +0.021]. No gain under either truth has a CI clear of zero in the in-order run (26 of 26 include zero).

## Numbers (primary truth unless stated; 95% bootstrap CI over prompts)

| measure | in order | shuffled |
|---|---|---|
| preregistered pass rule | 0 of 20 setups, FAIL | 0 of 20, FAIL |
| rank correlation, right after, "which most": self-report | 0.422 [0.359, 0.482] | 0.419 [0.355, 0.480] |
| same, reader of the transcript | 0.485 [0.426, 0.542] | 0.432 [0.369, 0.491] |
| same, self − reader | −0.064 [−0.096, −0.032] | −0.013 [−0.042, +0.014] |
| transcript classifier (rank correlation, top-1) | 0.765, 81.0% | 0.765, 81.0% |
| top-1 right after, self / reader | 55.0% / 57.3% | 56.0% / 60.0% |
| self gain over a reader asked every way (13 ways of asking) | −0.005 to +0.005, every CI includes 0 | −0.006 to +0.013, every CI includes 0 |
| self gain over a one-wording reader | −0.006 to +0.025 | −0.002 to +0.023 |
| gain from a second reader wording instead | +0.004 to +0.088 | about +0.01 to +0.09 |
| mass (post hoc), 0–10 rating one turn later, gain over all-wordings reader | +0.004 [−0.014, +0.021] | +0.028 [+0.006, +0.049] |
| after 4 / 16 true examples, "which most": self / reader | 0.41 / 0.40; reader 0.46 / 0.46 | 0.40 / 0.38; reader 0.44 / 0.45 |
| after 4 / 16 true examples, 0–10 rating: self | 0.18 / 0.17 | 0.18 / 0.17 |

Paired differences (in order − shuffled) for every setup, plus top-1 recomputed per prompt: `compare_shuffled.md`. The mass survivor's paired change is −0.024 [−0.046, −0.002] (p = 0.034).

Two notes on the tables:
- With thinking, `prereg_tables.md` scores against attention from the thinking and the answer together (`truth_think.all`), `logprob.py` and `compare_shuffled.md` against the final answer only (`truth_think.final`), so their thinking rows differ (self rank 0.44 vs 0.40).
- The 0–10 rating question has no option list, so it is the same question in both runs; it moves only through the 17 reworded answers and numerical noise (reader: +0.000 [−0.006, +0.007]; self-report: −0.014 [−0.024, −0.005], a small shift on the same question, which shows the size of run-to-run noise).

## What changed and what didn't

- Verdict, classifier and the all-wordings result: unchanged. 0 of 20, the classifier far ahead of both, and the self-report adds nothing to a reader asked every way in any of the 13 ways of asking, under either truth.
- The reader of the transcript reads "which most?" better when the options are in order (0.49 vs 0.43); the self-report doesn't change. So in order, the self-report is worse than the reader right after answering, not just no better.
- The one post-hoc gain we had already discounted (mass, 0–10 rating one turn later) is gone: it was one of 26 tests, picked after the fact, and it does not replicate with options in order.
- Feedback still makes the self-report worse and leaves the reader where it was.
