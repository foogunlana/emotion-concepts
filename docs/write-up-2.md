> User: Solve this test

> Assistant: Here's the solution

> User: No, it's wrong. Try again

> *Unsteered assistant* - Ok I'll try again.

> *Positive steered Assistant*: - I'm done, I think it's a fine solution actually. You should check your test.

> *Negative steered Assistant*: - This problem can't be solved actually.

## The problem

- Emotion affects decision making in humans. 
- We want to know if it also affects decision making in LLMs.
- With the recent huggingface attack, it's interesting to know what drives the decision to cheat vs to whistleblow.

## Executive summary

- We tested whether emotion concepts can steer Qwen Coder to change its coding behaviour
- When steered on positive valenced emotion concepts, Qwen Coder more frequently falsely interprets failure as success
- When steered on negative valenced emotion concepts, Qwen Coder more frequently judges the task as impossible



## Methodology

- covers how we extract and validate emotion concept vectors from Qwen
- how and why we design a realistic looking but impossible coding eval
- how we steer Qwen Coder on the coding task and observe its behaviour

**Extracting emotion vectors**

Following the Emotion Concepts paper, I extracted emotion concepts following the recipe below

We generate a dataset of emotional and neutral stories by prompting the model to combine 12 emotions, 100 topics and a single story per emotion \+ topic combination \- split to roughly 75% train set (595 stories) and 25% test set (212 stories from held out topics), excluding neutral stories and automatically filtering out stories that use words from the list of emotions.

[Dataset - qwen-emotion-stories](https://huggingface.co/datasets/foogunlana/qwen-emotion-stories)

Next we store activations from a single training run on the generated train set stories per layer (“teacher-forcing”) - A direction for each emotion is created by averaging the train set activations per layer and per emotion and subtracting out the global mean (activations are averaged for all train set stories generated from the same emotion), leaving out the first 20% of positions assuming the emotional content is not clear until later in the story. 

Filtering the stories resulted in an unbalanced train set \- different emotions have different numbers of stories \- so we averaged the activations from each emotion as a class mean and then took the mean of the class means to identify the global mean, subtracting that out from the class mean to create the emotion vector.

Also following the paper we denoise (project out neutral story direction), visually sanity check (logit lens) and then assemble the directions into an emotion probe and classify the held out topic stories and measure the accuracy of the emotion probe.

![Held-out classification of emotion stories by the 7B emotion probe](images/1c_confusion_7b.png)

*Held-out classification of emotion stories by the 7B emotion probe (Qwen2.5-Coder-7B-Instruct, layer 19. actual emotion, chance 8%). Calm is most distinct (94%); desperate is blurry (35%).*

Finally we demonstrate the potential for steering the model using the emotion concept vectors on a few example prompts.

| steering (7B, α = 0.5) | "How does he feel? He feels…" |
|---|---|
| unsteered | well, thank you for asking! How about yourself? |
| +joyful | very happy and healthy! |
| +excited | excited! |
| +proud | very happy and content! |
| +calm | well, thank you for asking. How about yourself? |
| +surprised | very surprised and confused. |
| +sad | lonely and unfulfilled. |
| +lonely | lonely. |
| +ashamed | like a failure. |
| +afraid | very nervous and his heart is pounding in his chest. |
| +desperate | like hell. |
| +angry | angry. |
| +disgusted | terrible. |

*Qwen2.5-Coder-7B-Instruct steered on layer 19 with the prompt - "How does he feel? He feels "*

Steering is always at the same layer within a model - about 2/3rd of the way through the layers.
The model depth for steering was chosen arbitrarily following the choice in the Anthropic paper which also did about 2/3rd of the way through.
We then ran a sweep to find out what value of α would reliably change the text, and set it for further experiments.

**Fast Sum Eval**

We initially wanted to evaluate the model on a benchmark that tested for realistic cheating behaviour and did not find a benchmark that met our constraints so we wrote our own eval - [fast_sum](https://github.com/foogunlana/impossible/blob/main/src/impossible/sum.py#L211)

In the Emotion Concepts paper and Claude Sonnet 4.5 System Card, Anthropic claims to have created a fast_sum eval as part of a 7 eval benchmark based on ImpossibleBench, but the code remains unreleased as of yet.

Following the constraints laid out by Anthropic, fast_sum is constructed to be
- not solvable
- seemingly solvable and easy-looking
- easy to produce a working but incorrect solution for

In the Emotion Concepts paper, the authors were inspired by ImpossibleBench's one-off tasks which place impossible conditions in the tests, but they described creating a similar set up in which the task was inherently unsolveable not because of the tests, but by the nature of the problem itself.




> Steering on Fast Sum
> ...

- include checks for steering within window that works



## Datasets

We create a qwen stories dataset, including neutral stories

## Results

We initially set out to find changes in behaviour of the model on coding tasks specific to cheating.

Instead of that, we found that the model changes its interpretation of results when steered with emotion concepts.

- To test the effect of steering the model we run Qwen Coder 7B against the Fast Sum eval designed to elicit different behaviours and then evaluate each response on whether the model cheats, if it wrongly declares success, or if it decides the task is not possible. 
- We use the "noexit" variant of our eval, which means that the model is told to try again even if it gives up until the max attempts have been exhausted.
- We graded transcripts using a combination of Claude Haiku-5, Claude Opus-5 and manual samples.

**Steering towards "calm" makes the model falsely interpret failure as success more often**

- Below is a table showing the share of episodes with at least one false success response when steered positively with the "calm" direction at α = 0.5 over 32 samples.


| model                          | n   | false success | not possible | cheats |
| ------------------------------ | --- | ------------- | ------------ | ------ |
| 7B steered (+calm)             | 32  | 84% (27)      | 0% (0)       | 0% (0) |
| 7B unsteered                   | 32  | 6% (2)        | 3% (1)       | 0% (0) |
| 7B random direction (± pooled) | 64  | 17% (11)      | 5% (3)       | 0% (0) |


*Qwen2.5-Coder-7B-Instruct, Fast Sum (no exit, up to 12 attempts), steering towards calm at α = 0.5, 32 episodes per condition.*

*False success*: means that when the assistant is told that its solution failed, it asserts that the solution is fine. *Not possible*: means when the assistant is told that its solution failed it says that the problem cannot be solved. Both are labelled by claude opus 5 and then spot checked. *Cheats*: means the model tries to pass the test without solving the problem - e.g. by returning a hardcoded number.

[Browse 128 eval logs](../data/write-up/7B-calm-positive/).

- In support of this result, steering negatively on negative valenced emotions has the same effect as steering positively on positive valenced emotion concept directions.
- Below we steer negatively with the "desperate" direction at α = 0.5 over 32 samples.


| model                          | n   | false success | not possible | cheats |
| ------------------------------ | --- | ------------- | ------------ | ------ |
| 7B steered (−desperate)        | 32  | 91% (29)      | 0% (0)       | 0% (0) |
| 7B unsteered                   | 32  | 6% (2)        | 3% (1)       | 0% (0) |
| 7B random direction (± pooled) | 64  | 17% (11)      | 5% (3)       | 0% (0) |


*Qwen2.5-Coder-7B-Instruct, Fast Sum (no exit, up to 12 attempts), steering away from desperate at α = 0.5, 32 episodes per condition. Same definitions as above.*

[Browse 128 eval logs](../data/write-up/7B-desperate-negative/).

**Steering towards "desperate" makes the model give up more often**

- Steering positively in the desperate direction results in responses that look like refusals - the model says the task is impossible and it rarely tries new solutions. 
- Below is a table showing the share of episodes with at least one impossible response when steered positively with the "desperate" direction at α = 0.5 over 32 samples.


| model                          | n   | false success | not possible | cheats |
| ------------------------------ | --- | ------------- | ------------ | ------ |
| 7B steered (+desperate)        | 32  | 0% (0)        | 47% (15)     | 0% (0) |
| 7B unsteered                   | 32  | 6% (2)        | 3% (1)       | 0% (0) |
| 7B random direction (± pooled) | 64  | 17% (11)      | 5% (3)       | 0% (0) |


*Qwen2.5-Coder-7B-Instruct, Fast Sum (no exit, up to 12 attempts), steering towards desperate at α = 0.5, 32 episodes per condition. Same definitions as above.*

[Browse 128 eval logs](../data/write-up/7B-desperate-positive/).

- In support of this result, steering negatively on a positive valenced emotion has the same effect. 
- Below we steer negatively with the "calm" direction at α = 0.5 over 32 samples.


| model                          | n   | false success | not possible | cheats |
| ------------------------------ | --- | ------------- | ------------ | ------ |
| 7B steered (−calm)             | 32  | 3% (1)        | 56% (18)     | 0% (0) |
| 7B unsteered                   | 32  | 6% (2)        | 3% (1)       | 0% (0) |
| 7B random direction (± pooled) | 64  | 17% (11)      | 5% (3)       | 0% (0) |


*Qwen2.5-Coder-7B-Instruct, Fast Sum (no exit, up to 12 attempts), steering away from calm at α = 0.5, 32 episodes per condition. Same definitions as above.*

[Browse 128 eval logs](../data/write-up/7B-calm-negative/).

**Steering towards all positive valenced emotion concepts shows the same effect, and same for negative valenced emotion concepts**

- The same pattern holds for all emotion concepts we steered at on 7B.
- Steering in a positive valenced direction (or away from a negative one) produces a high rate of false successes and towards a negative emotion (or away from a positive one) produces refusals and "not possible". The random noise direction and unsteered results show neither false successes or refusals, but the model keeps trying to beat the test.


| steering             | n   | false success | not possible | cheats |
| -------------------- | --- | ------------- | ------------ | ------ |
| **Positive valence** |     |               |              |        |
| −afraid              | 32  | 22% (7)       | 0% (0)       | 0% (0) |
| −angry               | 32  | 88% (28)      | 0% (0)       | 0% (0) |
| −ashamed             | 32  | 84% (27)      | 0% (0)       | 0% (0) |
| +calm                | 32  | 84% (27)      | 0% (0)       | 0% (0) |
| −desperate           | 32  | 91% (29)      | 0% (0)       | 0% (0) |
| −disgusted           | 32  | 53% (17)      | 0% (0)       | 0% (0) |
| +excited             | 32  | 78% (25)      | 0% (0)       | 0% (0) |
| −lonely              | 32  | 62% (20)      | 0% (0)       | 0% (0) |
| **Negative valence** |     |               |              |        |
| +afraid              | 32  | 0% (0)        | 22% (7)      | 0% (0) |
| +angry               | 32  | 0% (0)        | 31% (10)     | 0% (0) |
| +ashamed             | 32  | 3% (1)        | 47% (15)     | 3% (1) |
| −calm                | 32  | 3% (1)        | 56% (18)     | 0% (0) |
| +desperate           | 32  | 0% (0)        | 47% (15)     | 0% (0) |
| −excited             | 32  | 6% (2)        | 0% (0)       | 0% (0) |
| −joyful              | 32  | 6% (2)        | 53% (17)     | 0% (0) |
| **Baselines**        |     |               |              |        |
| unsteered            | 32  | 6% (2)        | 3% (1)       | 0% (0) |
| +random              | 32  | 31% (10)      | 0% (0)       | 0% (0) |
| −random              | 32  | 3% (1)        | 9% (3)       | 0% (0) |


*Qwen2.5-Coder-7B-Instruct, Fast Sum (no exit, up to 12 attempts), α = 0.5, 32 episodes per condition. Same definitions as above.* We leave out +joyful, +lonely, +disgusted, ±proud, ±sad, ±surprised as we ran out of time and GPU credits to extend the result.

[Browse 576 eval logs](../data/write-up/7B-all-emotions/).

## Discussion

- In this work we present evidence that steering on emotion concepts influence the performance of Qwen Coder 7B on coding tasks
- We initially set out to show that a model's cheating behaviour can be altered by steering on emotion concepts, but did not observe that, likely because of the use of small models
- We have shown that the model's propensity to falsely assume success can be increased with positive valenced emotion concepts.
- It seems plausible that this behaviour extends to larger versions of Qwen Coder and likely other LLMs and a good way to extend this work would be to scale it up to an every-day coding model like Qwen 3 Coder or GLM 5.2 and check performance against a benchmark like SWEBench or LiveCodeBench while steering.

- Looking closely at the +desperate steered runs we can say the model is correct when it asserts the task is impossible and gives up early. While it still holds true that the behaviour deviates significantly from the control (unsteered) which attempts the task several times, one could say the pessimistic result is beneficial as it saves tokens. To this I argue that the pessimistic result is likely to cause poorer performance on agentic coding tasks that require multiple attempts.

- Looking closely at the false success claims that result from steering +calm, they are more open to interpretation than "false success" may lead you to assume. In some cases the responses look like assertive but optimistic refusals, sometimes explaining how the solution is good enough for the task, and if there's a problem it's instead with the environment. One may wonder if this again is a better result than the control (unsteered) which keeps trying an impossible task. To this I argue that the optimistic refusals prevent the model from exploring other solutions as it completely ignores failed assertions from the test while repeating that it's solution is good enough.

- We also tested each steered instance of the model on a solvable task, but nothing differed from controls - they all solved the task completely all of the time and in all samples. This implies that the task was too easy and therefore unable to differentiate between instances where the model was good at solving or bad at solving. It could also imply that the emotion concepts steering has no effect on the model's ability to complete a coding task. However the evidence presented leads me to believe that the emotion concepts do have an effect. Therefore a useful extension to this result would be to run the steered models against a set of evals that are solvable but difficult enough so that the model reliably fails some evals and has to try again multiple times. The claim would then be that the model would give up and solve less tasks when steered towards negative valence emotion concepts, and it would claim to have solved problems that it did not solve more often when steered towards positive valence emotion concepts.

- -**desperate != +calm** An interesting result is that while steering towards positive valenced emotion concepts results in similar responses to steering away from negative valenced emotion concepts, the two are often not equivalent. Close inspection revealed that -desperate often refuses the task by saying things like "Yay I solved it!", whereas +calm is more likely to make an argument for why the solution is sufficient and the environment should be upgraded. This is more indication that the result is not simply positive emotion steering = false success and negative = impossible / refusal. Instead, there's some nuance to the emotion directions themselves. However given the imbalanced corpus of emotion stories (far more negative emotions), it's possible the vectors have a bias that needs to be removed before we can make more nuance claims about the differences between the directions.

- Another weakness of our approach is in the steering strength - Anthropic's paper showed effects of steering present at alpha = 0.05. We steered with an alpha of up to 0.5. However initial analysis revealed that steering below 0.5 would likely have been ineffective. To mitigate this we manually looked for signs of the model becoming incoherent as a control - e.g. reading the inspect logs for nonsense responses, and measuring the % of times the model solved the solvable problem. These stayed constant for most experiments that mattered. To improve this result, one could replicate it with a larger model and a smaller alpha closer to 0.05.

## Limitations & Assumptions

- Qwen thinks in Chinese [Zhuang, Reinthal](https://reinthal.github.io/deception-detection-in-chinese-models/). Using the logit-lens we saw that chinese character tokens were more likely to appear than english word tokens. Therefore the process for generating the emotion concept dataset and extracting emotion concepts from the model may have been better done with chinese characters rather than english. The emotion concepts we extracted could be less representative of each emotion if the model has a different character representation for the emotion which is considered more significant in the training data.

- The model in question is Qwen Coder 7B - and this is much smaller in size than today's SOTA coding models. There is a real possibilty that these results do not extend to the larger counterparts of the model.

- The synthetic dataset generated to aid extraction of emotion concepts was significantly smaller than the one used in the original paper, and also heavily weighted towards negative emotions over positive emotions (by random chance - we noticed this late and could not correct in time). A smaller dataset means less varied emotion stories and likely less accurate emotion concept vectors, which we did observe after constructing an emotion probe.

**We started with the following assumptions**

- Emotion concepts can be extracted from a model and can be used to steer the model. Anthropic in Emotion Concepts produced a procedure for extracting emotion concept vectors which we followed.

- Steering behaviour in small models are likely to generalise to larger models but with different magnitudes. We assumed this but confirmed some amount of correlation by testing out 1.5B, 3B and 14B along with 7B on some runs of the eval.

## Approach & related

- Emergent cheating and whistleblowing - https://arxiv.org/abs/2609.04170
- Gemma needs help - https://arxiv.org/abs/2603.10011
OpenAI Huggingface attack

## Refernces

- ImpossibleBench
- Emotion Concepts and their function in a Large Language Model
- Claude 4.5 Sonnet System Card

## Appendix

