# Report-time boost on single words: boosting the word vs its label vs both, in both label orders (Qwen3-32B)

Question: on the single-word lists (`results/privileged_words/`), boosting attention on word k only while Qwen3-32B reads the report question (arm C) raises P(report = word k's label) by +580% at ×10 when the labels and options are in order, but only +25% when they are shuffled. One explanation was that the boost works mainly through the label token "[Wk]" next to the word, and that the in-order alignment of label, position and option list makes that route much stronger. This run tests that explanation.

Design: the same 100 lists, seed 11, report question, prefill "[W", band of layers 21–53, doses ×¼ ×½ ×2 ×5 ×10, and arm C only (boost on the report-question queries only). The boost goes on one of three targets for item k: the word's token (`word`, as in the earlier runs), the three tokens of its label "[Wk]" ("[W", "k", "]") (`label`), or both together (`both`). Each target is run in both label orders. Specificity is the same as in `analyze_words.py`: the change in log P(report = boosted item) minus the mean change for the other seven items, averaged over the rotation. "%" means exp(specificity) − 1, with 95% bootstrap CIs over contexts. Attention mass is head- and query-mean mass at the report-question queries in layers 21–53, as a ratio to no boost (geometric mean over k).

Code: `privileged/privileged_words.py --conly --target {word,label,both}` (the default, `--target word` without `--conly`, behaves as before), `privileged/run_words_label.sh`, `launch/privileged_words_label.sh`, `privileged/analyze_words_label.py`. Pods hypha-prod-pwlabel-inorder and hypha-prod-pwlabel-shuffled, one A100-80GB each, $1.95 in total. Data: `data/{inorder,shuffled}/{word,label,both}/`, `results.json`, `tables.md`.

## Check
For the `word` target, the specificities match the committed runs exactly at every dose, in both orders. For example, at ×10 in order, +1.917 against +1.917 committed; shuffled, +0.220 against +0.220.

## Result

| order | boosted | ×5 | ×10 | mass on word / on label at ×10 |
|---|---|---|---|---|
| in order | word | +147% [+123, +175] | +580% [+492, +681] | 7.46× / 0.96× |
| in order | label | +61% [+55, +68] | +144% [+130, +160] | 0.89× / 6.64× |
| in order | both | +217% [+187, +252] | +674% [+581, +779] | 6.79× / 6.46× |
| shuffled | word | +12% [+5, +19] | +25% [+15, +35] | 7.56× / 0.96× |
| shuffled | label | +21% [+13, +30] | +36% [+22, +51] | 0.87× / 6.32× |
| shuffled | both | +22% [+11, +34] | +32% [+16, +51] | 6.65× / 6.14× |

Paired, ×10: in order, label minus word is −1.02 log units [−1.17, −0.87]. Shuffled, it is +0.09 [−0.04, +0.21]. In order minus shuffled is +1.70 [+1.56, +1.83] for word, +0.59 [+0.47, +0.71] for label and +1.77 [+1.61, +1.92] for both (all p < 10⁻⁴). Every boost reached a similar mass on what it targeted (6–7.5× at ×10), and the item that was not boosted stayed about flat (0.87–0.96×), so the targets were not mixed up. All doses are in `tables.md`.

## Reading
- The explanation tested here is wrong. In order, boosting the label is about four times weaker than boosting the word (+144% against +580% at ×10), although the label's attention rose as much as the word's did. Boosting both is only a little stronger than the word alone. So the large in-order effect runs mainly through attention to the word itself, not to the label next to it.
- With shuffled labels every target gives a small effect (+25% to +36% at ×10). Label and word do about the same there, so the label route is not what the in-order run strengthens: the order gap is +1.70 log units for the word and only +0.59 for the label.
- What explains the gap is that, in order, extra attention to a word converts into its label far more easily. The likely reason is that the word's slot in the list then equals its label digit and its place in the option list, so the report can follow which slot was attended to. With shuffled labels the model would have to bind that particular word to an arbitrary label, and that binding barely carries the boost. The earlier check showed that the shuffled report does not follow the slot either, so the slot only helps when it matches the label. This last point is an inference from the pattern; this run did not test it directly.
