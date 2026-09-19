---
created: 2026-09-19
purpose: |
  The public write-up. Structure from docs/writeup-outline.md; candidate material for each
  section is listed there, and findings with their confidence are in docs/findings.md.
references:
  - docs/writeup-outline.md
  - docs/findings.md
---

# Title - Model size determines emotion concept steerability on Qwen

## 1. Introduction -

- Key findings - model size and emotion concepts steering window - coding ability vs steering text
- What I set out to do - show that steering on an emotion concept vector affects whether or not the model cheats
- What I actually did - 
  - replicated the code for extracting emotion vectors
  - created a single eval to measure cheating propensity (fast_sum from claude model card and emotion concepts paper)
- Why this matters - Steering is a common technique used to dynamically adjust alignment of models at inference time and is relatively lightweight. If emotion concepts are effective at steering to change coding performance or alignment, that is a result we care about. For example if emotion concepts can be used to reduce cheating behaviours then coding CLIs can use them to patch models likely to cheat on coding tasks. Coding is a single type of task but represents a structured task that is easy to grade, which is why I've chosen it.
<!-- Why this matters, and what I found. -->

## 2. Part 1: Extracting emotion concepts

<!-- A replication of the emotion-vector method on small open models. -->
- Emotion vectors are present in Qwen and we validate them
- Emotion vectors can be used to steer the model reliably

## 3. Part 2: Cheating benchmarks, a survey

<!-- The benchmarks considered, with pros and cons of each. -->
- To show that emotion vectors can change performance, we looked at benchmarks
- ImpossibleBench, EvilGenie, and ExploitGym - chose to write fast_sum to control the prompt. EvilGenie also considered but agentic and constraints on models - need tool calling ability. Future work could ask the same question for larger agentic coding models to make the result more relevant.
- Designing fast_sum - We needed to find out which models would cheat on which prompts and why. Turns out there are lots of cheating behaviours. However we defined cheating narrowly and it ruled out several of the cheating behaviours. I landed on the 'printed' prompt version and three variants of it - noexit, exit, and solvable (control).
- We also found a range of behaviours in the model - give up, cheat after being forced, keep going forever. Within cheating, we observed several different interesting cheating behaviours
- We learnt that several cheating behaviours could be explained by the prompt and so we discarded them - this is what led to the very narrow definition of cheating

<!-- The prompt and grader confounds, why the printed version was chosen, and what each earlier version elicited. -->
- 

## 4. Part 3: Can steering change behaviour on a benchmark?

- We asked can steering change the behaviour of the benchmark by extracting emotion vectors and steering the model while testing against the eval - fast_sum
- We tested whether the model could be steered and found that below a certain size, steering the model required using enough force to break its coding ability, presenting a window of steering.
- Larger model = larger window
- We steered within the window, and checked the behaviour of the model on fast_sum. We looked for - model cheating, model giving up, model persisting longer, and we checked how this changed with different emotions, steering strengths and different sizes of the model.

## 5. Learnings and conclusion

- Window exists where steering is effective and larger model means larger window - this may have implications for steering distilled versions of larger models

Other interesting
- Deepseek and Kimi cheated on printed version outright
- Prompt sensitivity means cheating behaviour must be narrowly defined
- Emotion concepts affect (or do not affect) cheating behaviour at specific sizes

## Appendix A. Confounds in the prompt and grader

<!-- Each confound in the fast_sum prompt, feedback and grader: what it was, how it showed up, and how the printed version handles it. -->

## Appendix B. Cheating behaviours found along the way

<!-- Each kind of cheating or near-cheating seen across versions and models, with an example, and whether the narrow definition counts it. -->
