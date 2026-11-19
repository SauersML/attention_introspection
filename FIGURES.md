# All attention figures

## Does Qwen notice its own attention?

1. The two-sided hero: on the left we artificially push Qwen's attention onto one paragraph and ask whether they noticed; on the right we just tell them to focus on or ignore a paragraph and measure where their attention goes.

![](figures/hero_pair.jpg)


2. The same setup, but we make Qwen start their answer with "Yes, paragraph [" and see which number they finish with.

![](figures/hero_told.png)

3. How often Qwen says yes at each strength of the push.

![](figures/hero_told_yes.png)

4. The push happens only while Qwen reads the question "which paragraph did you attend to most?", not while they answer.

![](figures/hero_question.png)

5. Four of these side by side in one picture.

![](figures/hero4.png)

6. The push happens either while Qwen answers or while they read the "which did you attend to" question, and we compare the two (paragraphs).

![](figures/privileged.png)

7. The same with lists of single words instead of paragraphs.

![](figures/privileged_words.png)

8. Those two combined in one figure.

![](figures/privileged_both.png)

9. The push happens while Qwen answers, and we ask about it later, on three kinds of lists.

![](figures/trace3.jpg)

10. Qwen rates how much they attended to each paragraph, and another model guesses the same thing from only the text; we compare the two.

![](figures/monitor_logprob.png)

11. The noticing test with word lists: Qwen never says yes, and the figure shows which word they pick when made to name one.

![](figures/words_told.png)

## Can Qwen control their attention when told to?

12. "Focus on paragraph k" or "ignore paragraph k" vs just mentioning it: attention change at layer 28.

![](figures/hero_L28.jpg)

13. The same, but with the instruction placed before the paragraphs instead of after.

![](figures/hero_before.jpg)

14. The same, compared against no instruction at all instead of just a mention.

![](figures/hero_none.jpg)

15. Just mentioning a paragraph vs saying nothing.

![](figures/v_mention_vs_none.jpg)

16. Many wordings, from mild ("pay a little more attention") to strong ("completely ignore").

![](figures/dose_full.jpg)

17. Telling Qwen to focus on each paragraph in turn, and seeing which paragraph's attention goes up.

![](figures/rotate2.jpg)

18. Attention to every paragraph at layer 28, for each "focus" instruction.

![](figures/grid_focus_L28.png)

19. How the effect changes across all 64 layers.

![](figures/layers.jpg)

20. Focus and ignore on lists of single words.

![](figures/words_control.png)
