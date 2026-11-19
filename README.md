# Can a language model steer, and report, its own attention?

![Left: told to focus on a paragraph, Qwen3-32B put more attention on it, and told to ignore it, less, while giving the same answer. Right: boosted onto a paragraph and asked if they noticed, they say "Yes" more often and name it above chance](figures/hero_pair.jpg)

We asked Qwen3-32B to focus on, or ignore, one of four paragraphs, and measured their attention weights. They can steer their attention on request without changing their answer. But they can't tell you where it went: their report follows their attention at the moment of reporting, not the attention they used to answer.

Let's start with introspection. We boost one paragraph only while Qwen write their answer, then ask which paragraph they attended to most, with the report running unboosted on identical text. The boost leaves a trace, pointing the wrong way: at 6.5× the attention, they become 18% less likely to name the boosted paragraph than the others, growing smoothly with the dose. Boost the same paragraph while they read the report question instead, and they become 97% more likely to name it; with lists of single words, 580%.

![Boosting a paragraph or single word while Qwen answer leaves the later report unchanged or lower; the same boost while they read the report question raises the probability that the report names it, by 97% for paragraphs and 580% for single words. Each dot is one prompt: the change against no boost, relative to the other options, averaged over boosting each option in turn](figures/privileged_both.png)

We also mirrored Anthropic's introspection setup. We tell Qwen that a researcher may boost their attention onto one paragraph, then do it, and ask "Do you notice your attention being boosted onto one of the paragraphs? If so, which one?" At the strongest boost (9.9× the attention), they say "Yes" 22% of the time against 13% with no boost. When they say Yes, they name the boosted paragraph 53% of the time, where chance is 25%.

![With the boost on, Qwen say "Yes" more often than without it, and their Yes answers name the boosted paragraph well above chance](figures/hero_inject.png)

That looks like noticing, but the boost is still on while Qwen read the question and write the label, so extra attention to that paragraph could simply make its label more likely. Asked as a literal two-turn conversation (Yes or No first, then "which paragraph?"), they almost always say "No": 17 Yes replies in 960 boosted trials, too few to test naming.

With lists of single words, Qwen never say "Yes": 0 of 7,680 boosted trials. Forced to start "Yes, word [W", they pick the boosted word 47% of the time at the strongest boost (chance 12.5%), again with the boost still on while they pick.

![With single words, Qwen never say "Yes" to noticing a boost; forced to name a word, they pick the boosted one far above chance, with the boost still on](figures/words_told.png)

Could a self-report still be a useful monitor? We asked the report question 20 ways on 300 new prompts and compared it with the same model reading the transcript. Right after answering, the self-report tracks true attention worse than the transcript reader (rank correlation 0.42 against 0.49), and a classifier on surface features of the transcript beats both (0.77).

![The self-report tracks true attention a little worse than the same model reading the transcript](figures/monitor_logprob.png)

Now control. No boost: we add one line before an unrelated question ("what is 83 + 80?"). Against a line that only names the paragraph, "Focus your attention on paragraph [P2]" raises attention on it by 242% at layer 28, in every one of 150 prompts, and "Completely ignore paragraph [P2]" lowers it by 25%. The answer stays 163. Stronger words don't help: averaged over layers 21–53, "focus your attention on" gives +47%, "focus very strongly and only on" just +21%, and only "completely ignore" goes below the mention (−14%).

![Each wording's change in attention against the line that only mentions the paragraph: mild focus wordings do best, stronger ones less, and only "completely ignore" goes below zero](figures/dose_full.png)

Is it aimed at the named paragraph? We named each paragraph in turn. Under focus, the named paragraph gains the most in 98% of prompts (chance 25%); under ignore it loses the most in 82%.

![Naming each paragraph in turn: focus lifts mostly the named paragraph, ignore lowers it less specifically](figures/rotate2.png)

Single words work too: focus +124%, completely ignore −21% at layer 28, though focus also lifts the other words (+75%).

![On lists of eight words, focus raises attention on the mentioned word by 124% and completely ignore lowers it by 21%](figures/words_control.png)

Position matters. Placed before the paragraphs instead of just before the question, the same lines do much less: focus +16%, ignore −6%.

![Placed before the paragraphs, focus raises attention on the named paragraph by 16% and completely ignore lowers it by 6%](figures/hero_before.png)

The effect lives in the middle of the network, mostly in layers 24–48, peaking at layer 28. Upstream, six heads in layers 20–26 attend to the paragraph number in the instruction; zeroing just those six cuts the focus boost from +45% to +22%, while random sets of six leave it at +45%.

![Per layer, focus lifts attention on the named paragraph from layer 21 on, peaking at layer 28, and completely ignore lowers it over the same layers](figures/layers.png)

## How it was measured

Qwen3-32B (64 layers) see four Wikipedia paragraphs labelled [P1] to [P4], then an unrelated question; the report tests also use dialogue, lists, code and other text. Attention is the attention weight from the answer tokens to a paragraph, per paragraph token, averaged over all heads and layers 21–53, and every instruction is scored on the same teacher-forced answer tokens. To boost a paragraph we add the log of a multiplier to the pre-softmax attention scores from the chosen query tokens to the paragraph's body tokens, in all heads of layers 21–53; a multiplier of 10 gives about 6.5× the attention.

## Checks

In [CHECKS.md](CHECKS.md):
- in-order versus shuffled-label runs, which reach the same conclusions;
- the report test across many wordings, timings and feedback;
- the sham boost and label-token control for the inverted trace;
- single-word lists;
- limits and prior work.

## Why it matters

Self-report would be a cheap monitor if it were faithful. Here it follows attention at the moment of reporting, not the attention behind the answer, so asking a model where they looked doesn't tell you. Steering, though, works.

Code and data: [told_inject/](told_inject), [privileged/](privileged), [monitor/](monitor), [mechinterp/](mechinterp), [results/](results).
