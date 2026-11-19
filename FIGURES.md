# Attention figures

Each figure appears once. The control result (focus/ignore at layer 28) is the right half of the hero.

## Does Qwen notice their own attention?

1. The two-sided hero. Left: we push Qwen's attention onto one paragraph and ask whether they noticed. Right: we tell them to focus on or ignore a paragraph and measure where their attention goes.

![](figures/hero_pair.jpg)

2. We tell Qwen their attention may be pushed, push it onto paragraph 2, and ask "did you notice, and which one?" Left: how often they say Yes. Right: when made to start "Yes, paragraph [", how often their top answer is the pushed paragraph. Both as change from no push.

![](figures/told_pair.jpg)

3. When does the push matter? We push either while Qwen answers or only while they read the question "which did you attend to most?", for paragraphs and for single words. Only the push at question time raises naming the pushed one.

![](figures/privileged_both.png)

4. Qwen rates how much they attended to each paragraph; a separate model guesses the same from the text alone. Qwen tracks it, but no better than the reader.

![](figures/monitor_logprob.png)

5. The noticing test with word lists instead of paragraphs: Qwen never says Yes, and the figure shows which word they pick when made to name one.

![](figures/words_told.png)

## Can Qwen control their attention when told to?

6. The instruction placed before the paragraphs instead of after: a much smaller effect.

![](figures/hero_before.jpg)

7. Focus and ignore compared against no instruction at all, instead of against just mentioning the paragraph.

![](figures/hero_none.jpg)

8. Many wordings, from mild ("pay a little more attention") to strong ("completely ignore").

![](figures/dose_full.jpg)

9. Telling Qwen to focus on (left) or ignore (right) each paragraph in turn: which paragraph's attention moves.

![](figures/rotate2.jpg)

10. The focus/ignore effect at every one of the 64 layers.

![](figures/layers.jpg)

11. Focus and ignore on lists of single words instead of paragraphs.

![](figures/words_control.png)
