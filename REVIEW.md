# Review fixes, 18 Nov 2026

Two reviews: a code review of this repository (findings 1–6), and an independent source review by Vesper (`hypha-experiments-1118.md`, findings A1–A4; thanks to Vesper). Each finding below, then what changed. New GPU runs: two small Qwen3-32B pod runs (A100, $1.11 in total, plus $0.16 for two first attempts stopped before loading the model, one out of disk space).

## 1. Monitor: "private signal" claim (with Vesper A1, A2)
Finding: the shuffled self-report is the wrong null (it breaks within-prompt structure and every link to the transcript, so any second noisy reader beats it); the headline used total mass, not the preregistered length-normalised truth; the pass rule verdict was not reported; feedback examples showed the normalised truth.
Changed:
- `monitor/logprob.py` now runs both truths (norm = preregistered primary, mass = secondary, post hoc), and for each way of asking adds the self-report to a base with the reader's scores for every wording asked at that timing (+ position, length), and, as a second null, measures what a different reader wording adds to the one-wording reader. Gain CIs use 2,000 bootstrap resamples (was 200). The shuffle is kept as a marginal control only, reported as exceedance counts b of 200 with p = (b + 1)/201.
- Result (as the review expected): right after answering the self-report adds −0.005 to −0.001 to the all-wordings reader (mass) and −0.006 to +0.004 (norm), CIs over 0; only "0–10 rating, one turn later" under mass has a CI clear of 0, +0.028 [+0.006, +0.049], and is reported as one post hoc pick of 26. A second reader wording adds as much or more than the self-report.
- Ran the preregistered analysis (`monitor/analyze_monitor.py`) on all 300 prompts: 0 of 20 setups pass, verdict FAIL; tables saved as `results/monitor/prereg_tables.md`. Right after, primary truth: self 0.419 vs reader 0.432.
- `results/monitor/LOGPROB.md` rewritten: leads with the primary truth and the FAIL verdict, mass labelled post hoc, says the feedback examples used the normalised truth (so the feedback drop is not a target mismatch), and notes the demo-bank dependency Vesper raised (examples from the same family and shard; a prompt's own label never in its own prompt).
- README paragraph and summary rewritten; `figures/monitor_logprob.png` redrawn on the primary truth, with the all-wordings-reader gain and no shuffle band or p values; the reader is orange instead of grey.

## 2. Ignore specificity
Finding: report ignore as named-minus-other; against no instruction it is not specific; the ignore line is longer than the mention.
Changed: `results/qwen3-32b-rotate2/named_minus_other.py` (+ `.json`): against the mention line the named paragraph loses 8.9 points more than the other three [8.6, 9.2]; against no instruction it loses the most in only 23% of cases (chance 25%) and every ignore line lowers P4 by about 20%. README states both. Ran the length-matched mention (16 tokens, same 150 contexts as the dose run; `results/qwen3-32b-mention16/`, pod hypha-prod-attnmention16b, $0.59): the original conditions reproduce exactly (ignore vs mention −14.3%, −24.9% at layer 28); ignore vs the 16-token mention is −16.4% [−17.3, −15.5] (−30.1% at layer 28), so the ignore effect is not from the longer line.

## 3. Report test numbers
Changed: README names the truth definition and gives both: last prompt token, self 58% / reader 63% (classifier 71%); mean over answer positions, self 57.3% / reader 48.0% (classifier 75.8%).

## 4. Sham null
Finding: the header/question sham does not depend on the target paragraph, so its paragraph-specific shift is zero by construction.
Changed: reworded in README and `ROTATE.md` (correction note). Ran the label-token sham (bias on the "[Pk]" tokens of the same paragraph, same 100 contexts, b = 2 and 4; `results/qwen3-32b-rotate/sham_label.py`, pod hypha-prod-attnshamlabelb, $0.52): body boost −0.54 [−0.73, −0.37] (reproduced), label boost +0.13 [+0.03, +0.22] and +0.43 [+0.23, +0.64]. The inverted trace belongs to attention on the paragraph text; attention on the label pushes the report the other way. Written up in `ROTATE.md`, README, and the overturn line of `results/privileged_shuffled/RESULT.md`.

## 5. Privileged figure and README (with Vesper A3)
Finding: "×10" is a log-10 pre-softmax bias (≈ 6.5× attention mass; ×5 ≈ 3.9×); the effect is relative to the other labels, not a target-only probability change; arm C differs by kind; per-kind arm A range; single words have no answer-time trace.
Changed: `privileged/target_only.py` (+ `results/privileged_shuffled/target_only.json`) tabulates the target label's own change beside the relative measure: arm A at ×10 −10.1% relative, −4.6% target alone; arm C +91.4% relative, +74.3% target alone; arm A per kind at ×10 from −1.0% (lists, CI includes 0) to −25.2% (paragraph QA); arm C at ×10 paragraph arithmetic +112% vs paragraph QA +11.5% [−6.5, +34.7]. README and `RESULT.md` verdict state the units and both measures. `figures/privileged.png`: dose ticks now show measured attention mass (×0.27 … ×6.5), subtitles ×6.5/×6.6, side labels say "than the others", bars for other labels black instead of grey, p text larger. README says single-word lists show no answer-time trace.

## 6. Minor
- Token-grid README text described an older figure (four prompts, a "do not" condition); now describes the current one (two prompts, focus naming each paragraph in turn, layer 28, scaled per prompt).
- 97% vs 98%: README now names both baselines (98% against the mention line, 97% against no instruction).
- "Layers 0–20 ≤16%": false for "twice" (+25% at layer 18); fixed in README and `DOSE.md`.
- `figures/trace3.png`: y label now says the change is relative to the other paragraphs; data paths made repository-relative.
- Two estimates at the same dose (−0.36 on 200 prompts, −0.54 on 100): README says both and why they differ.
- "Answer text stayed exactly the same": now "all 200 at the smaller boost, 199 of 200 at the larger".
- First token of each paragraph outside the span: stated in README, `DOSE.md` and the `hero_L28.py` docstring.
- `hero_L28.py`: docstring describes the exact query positions; data path is repository-relative (added `figures/layers3_data.json`, which `layers3.py` writes); figure re-rendered pixel-identical.
- `logprob.py`: bootstrap of the gain 200 → 2,000; shuffle p = (b + 1)/(N + 1).

## Vesper A4: query positions differ between experiments
Changed: README and `DOSE.md` state the convention. The dose run and the headline use the positions occupied by the forced answer tokens (`att_mean`); the last prompt token is stored separately (focus +6.4% there vs +47.3%; very strongly and only −6.2% vs +20.9%). `monitor.py` uses the predicting positions; the privileged run uses last prompt token through last answer token. The mean over predicting positions was not saved for the dose run and is not substituted. Also from A4: README no longer says "do not pay attention" lowers attention against no instruction (+8.5% in the dose run).

## Figure review, 18 Nov (hero_before, privileged)
- `figures/hero_before.png`: violins now sit on a labelled y axis (change in attention on [P2]); the zero line is named as the line that only mentions the paragraph; mean and 95% bootstrap interval drawn on each violin (focus +16% [+14, +18], completely ignore −6% [−8, −3]); dots one colour per instruction; one ignore prompt at +192% shown above a break in the axis instead of stretching it.
- `figures/privileged.png`: dots coloured by arm only (blue while answering, orange while reading the question) instead of a value colour scale; 95% intervals drawn as black boxes with the mean as a white line, visible on every violin; y ticks at ×0.5, ×1, ×2, ×4, ×8 (−50% to +700%), evenly spaced on the axis, which is marked "log scale"; p shown at the largest boost and on the three settings with p ≥ 0.0001 (all reading-question boosts up to ×1.8).
- No sham run on non-paragraph tokens: the trace measure is paragraph-specific (boosted minus the other paragraphs, every paragraph boosted in turn in the privileged run), so a boost that does not depend on the target cannot shift it, and the label-token sham (section 4) is the target-dependent control. README says so.
