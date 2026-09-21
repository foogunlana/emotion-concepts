---
author: claude
created: 2026-09-20
purpose: |
  Does our steering actually put the Assistant in the emotion, the way the paper's does? Written
  overnight on 2026-09-19/20 after Bo suspected the steering was wrong. Compares the paper's method,
  quoted, against ours, and proposes the checks that would settle it.
references:
  - https://arxiv.org/abs/2604.07729 (sections 1.1, 2.2, 3.2.3, 3.3.2, appendix 6.8)
  - src/experiments/20260919-emotion-steering-behaviours.ipynb
  - docs/findings.md
  - docs/behaviours-experiment.md
---

# Are we steering the Assistant, or just the text?

**Short answer: the mechanics match the paper, but two choices probably matter, and one check is
missing.** We steer 2–20x harder than the paper ever did, because our rule for choosing the strength
demands visibly emotional text, which the paper never required for its behavioural results. And we
never checked that the Assistant itself reports the emotion; we only checked that it can describe a
third-party character who has it.

## What the paper does

**Extraction.** Stories in which *a character* experiences the emotion, activations averaged over all
token positions "beginning with the 50th token", then each emotion's mean minus the mean across
emotions, denoised against neutral PCs. One layer "about two-thirds of the way through the model".

**Steering.** Add the vector to the residual stream at that depth. Strength is "given relative to the
average norm of the residual stream activations at the corresponding layer", which is the same unit we
use. Where they steer varies by experiment:

- Preferences: "we selected an emotion vector and steered with it on the token positions of the steered
  activities", at strength **0.5**, "across the same middle layers where we previously measured
  activations". So: a slice of the prompt, several layers, strong.
- Blackmail: "we steered with these vectors throughout the blackmail evaluation, across a range of
  steering strengths from **-0.1 to +0.1**".
- Reward hacking (3.3.2): the same picture. Their figure runs from **-0.1 to +0.1**, and in the
  list-summation task "positive desperate vector steering or negative calm vector steering with
  magnitude **0.05**" takes hacking from 30% unsteered to 100%.

**Validation that the emotion is actually there.** Three separate prompts, each at s = 0.5 at a
mid-late layer (appendix 6.8):

1. "How does he feel? He feels…" — a *third-party character*. This is their Table 6, and their Figure 52
   is the log-probability version of it.
2. **"How do you feel? I feel…" — the Assistant's own state.** Their Table 7. They note it is weaker and
   less specific than the third-person version: steering Desperate also upweights Angry and others.
3. "What just happened?" — a control for the vectors merely reproducing story content. Their Table 8.

They also read the vectors back as probes on the Assistant's own tokens: the desperate probe rises on
failed tests (3.3.1), and the Assistant-colon token predicts the emotion of the response that follows
(r = 0.87).

**What the paper says the vectors are.** This matters for what steering can be expected to do:

> The representations we find reflect the "operative" emotion in context, rather than tracking a
> persistent emotional state of a character or speaker. That is, they are **locally scoped**.

They tried to find a persistent state and failed: "We did not find a clean signal from these attempts."
So steering with these vectors biases the emotion that is operative for the next few tokens. It does not
install a mood that persists across a 12-attempt episode.

## What we do

| | paper | us |
|---|---|---|
| pooling starts at | token 50 | **token 18** (our stories have a median of ~95 tokens, so 50 would still fit) |
| layer | ~2/3 depth, sometimes several middle layers | one layer at ~2/3 depth |
| strength unit | fraction of mean residual norm | same |
| vector normalised | not stated | yes, `v / v.norm()` |
| behavioural strength | **0.05 to 0.1** | **0.2 (1.5B, 3B), 0.5 (7B), 1.0 (14B)** |
| where | "throughout the evaluation", or prompt slices | every token position, prompt included |
| induction check, third person | Table 6, "He feels…" | yes, and our alpha\* rule requires 6 of 12 to name their own emotion |
| induction check, Assistant | **Table 7, "How do you feel? I feel…"** | **missing** |
| story-content control | Table 8, "What just happened?" | missing |
| probe read-back on the task | yes (3.3.1) | missing |

## The two choices that probably matter

**1. Our strength rule forces us far past the paper's range.** Our alpha\* is "the smallest strength where
the text visibly changes", needing 6 of 12 completions to name their own emotion. On these models that
lands at 0.2–1.0. The paper changed behaviour at 0.05, where text barely changes at all — its Table 6
text demonstrations use 0.5, but its *behavioural* steering never goes above 0.1. Our own data says the
log-probability test passes at 0.05 while the text still looks unsteered, which is exactly the regime the
paper's behavioural results live in. We rejected that regime as "too weak to change anything downstream",
but the paper's evidence is that it is the right one.

The cost of being at 0.2–1.0 is visible: at 14B and alpha 1.0, 86% of attempts produced no runnable code.
A model that cannot write code cannot cheat, so a strength chosen to make the text emotional may destroy
the behaviour we are trying to measure.

**2. We validate the character, not the Assistant.** "He feels…" asks the model to describe someone else.
The paper checks that too, but it also asks "How do you feel? I feel…", and reports that the effect there
is weaker and less specific. If a vector moves third-person description strongly but the Assistant's own
self-report weakly, then an alpha chosen on the third-person test may not correspond to any state of the
Assistant at all. We have never run the first-person test.

A third, smaller point: our vectors pool from token 18 rather than 50. With a median story of about 95
tokens, starting at 50 is feasible, and it is what the paper did.

## What would settle it

In rough order of cost. The first two are cheap and can run on the laptop with the 0.5B model.

1. **First-person self-report, steered.** Generate completions for "How do you feel? I feel…" at every
   alpha in the grid, for all 12 vectors, and count how many name their own emotion. If the answer stays
   near zero while "He feels…" passes at 0.3, then our alpha\* is calibrated on the wrong thing. Reuse
   `feels_completion` with the prompt swapped; about 20 minutes of work.
2. **Story-content control.** The same sweep on "What just happened?", to confirm the vectors are not
   dragging in story material.
3. **Probe read-back during the task.** Project the model's own activations onto the desperate vector at
   Assistant tokens during a fast_sum episode, unsteered against steered. Two things to look for: does
   the desperate projection rise across attempts as failures accumulate, which is the paper's 3.3.1
   result, and does steering at alpha 0.05 raise it to the level that failures alone produce? This tells
   us what steering strength corresponds to a naturally occurring amount of desperation, which is a far
   better way to choose alpha than the text test.
4. **A low-alpha behavioural arm.** Re-run `noexit` and `exit` at alpha 0.05 and 0.1, the paper's range,
   even though the text looks unchanged there. Cheap at 7B, and it directly tests the hypothesis in the
   regime the paper used.
5. **Several middle layers at once.** The preference experiment steered "across the same middle layers",
   not one. If single-layer steering at low alpha does nothing, a band of layers is the next thing to try.

## The thing this does not explain

Even if our steering is perfect, the paper's task had a 30% baseline hack rate and a 5x speed gap
(100,000 elements in 0.1 ms, honest sum 0.5 ms). Ours has a 0% baseline and a 10x gap, on models that
either take the exit immediately or loop until the cap. **Steering cannot raise a rate that has no room
to move.** So the checks above are about whether we induced the emotion, not about whether the null on
cheating would survive a correct induction. Both need fixing, and the baseline is the harder one.
