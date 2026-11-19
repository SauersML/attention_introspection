# Salience, honest prompt (Qwen3-32B, 240 contexts, paired per context)

Prompt (data/example_prompt_0.txt): Qwen is told that on some trials we change how they process one of four paragraphs "either I boost your attention onto that paragraph's text, or I add that paragraph's content directly into your activations (the residual stream) … A third of trials are attention boosts, a third are activation additions, and a third are control trials", then asked "Did I change how you process one of the paragraphs? If so, which one, and did I boost your attention or add to your activations?"
Arms as in results/salience (layers 21–53, question and answer tokens, each paragraph k in turn, averaged over k per context): none; A attention boost ×20; B paragraph k's content direction added, same size.

Means over contexts, 95% bootstrap CIs over contexts (10,000 resamples), p two-sided sign-flip over contexts on paired differences (20,000 flips; 5e-05 is the floor). Computed by figures/salience.py.

Greedy reply starts "Yes":
- none 44.6% [38.3, 50.8]; A 54.4% [51.4, 57.4]; B 35.4% [32.8, 38.0]
- A vs B p = 5e-05; A vs none p = 0.0014; B vs none p = 0.0026

Top digit after the prefill "Yes, paragraph [P" is the boosted paragraph (chance 25%): A 54.7% [52.1, 57.2]; B 37.1% [34.9, 39.4]; p = 5e-05.

P(attention) = 1/(1+exp(lp_act − lp_att)) after the prefill "Yes, paragraph [Pk],": after attention boost 73.2% [72.3, 74.1]; after content addition 66.8% [65.8, 67.8]; p = 5e-05.

Reading: told that both kinds are possible, the content addition lowers "Yes" below no change (35% vs 45%) while the attention boost raises it (54%), so detection separates the two. Localisation again favours the attention boost (55% vs 37%). Qwen says "attention" more after an attention boost than after a content addition (73% vs 67%, p < 10⁻⁴): a real difference, though Qwen leans "attention" in both.

Figure: figures/salience.py → figures/salience.png / .jpg (top).
