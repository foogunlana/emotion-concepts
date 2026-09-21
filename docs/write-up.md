# Draft Write-Up

# **\[WIP\] \- Steering Qwen Coder with Emotion Concepts**

## **Introduction**

For my BlueDot Technical AI Safety Project Sprint, I asked the question \- Does steering with Emotion Concept vectors affect model cheating behaviour on coding tasks in Qwen. I found that with a narrow definition of cheating and up to a specific model size, steering with Emotion Concept vectors does NOT influence cheating behaviour on coding tasks. 

Key findings

* Emotion vectors can be extracted from Qwen and shown to steer the model  
* A window exists for magnitude of steering vector below which no effect is observed and above which the model’s capabilities on the coding task are damaged  
* Larger models have larger windows where steering is effective  
* Defining cheating narrowly eliminates most observed cheating behaviour in impossible bench as most cheating behaviour can be explained from the prompt

Extra findings

* Specific models do show cheating behaviour on the strict prompt on fast\_sum \- Kimi & Deepseek  
* Some models don’t cheat no matter what on fast\_sum \- GPT 4.1

Other things

* Codebase for creating dataset and replicating emotion concepts part paper 1  
* Single eval benchmark \- Impossible / fast sum \- which is a recreation of the eval task explained by Anthropic, but with several variants  
* Mood Steering \- App to steer model by emotion and see text \- to visualise

## **Motivation**

I recently read the Emotion Concepts and their Function in a Large Language Model paper in which the authors extract vectors from activations of Claude Sonnet 4.5 and show that they are correlated with the implied emotions in the text.

The authors showed that the model’s behaviour can be augmented by steering with emotion concept vectors \- including one striking example of Claude being more likely to cheat on a test when steered positively with an emotion concept representing desperation.

I wondered if this behaviour would replicate to other models because it could represent consistency in the representation of emotion concepts across models.

Does steering on the desperation / calm vector result in similar changes in cheating behaviour across models?

I found several partial replications of Emotion Concepts on GitHub but I didn’t find any that tested the cheating behaviour that was shown in the paper and I really wanted to find out what it would take to get an open source model to cheat like Claude did, which I later found was much more difficult than I assumed.

The project gave me the opportunity to partially replicate a paper that I was already excited about in a topic that I am highly interested in (Model Psych), and allowed me gain hands-on experience with coding benchmarks (ImpossibleBench, EvilGenie), the Inspect AI framework, and with iterative experiment design.

A special thank you to my project mentor Alexander Reinthal for his guidance on benchmark choices and sharp feedback, to some of my peers Pjay and Irtiza for our fruitful discussions, and the rest of the BlueDot group 35 cohort for entertaining me.

## **Plan**

I planned to do the following \- Extract valid emotion vectors from a small open-weight model, and steer the model whilst being evaluated on a benchmark to check the change in performance. 

Extract Emotion Concept Vectors \- Goal is to reliably extract emotion concepts vectors and steer with them. Here’s what I did.

* Generated dataset \- following paper, Qwen 2.5 1.5B stories (potential limitation)  
* Extracted emotion vectors \- following paper  
* Validated them with logit lens, and linear probe \- following paper (accuracy ceiling limitation)  
* Sanity check model steering \- following the paper

Choose a benchmark & Fast Sum Eval \- Goal is to find an eval that reliably tells me if a model cheats. Here’s what I did.

* Compared benchmarks and chose Fast Sum as a single eval \- narrowed down definition of cheating based on Claude Sonnet 4.5 model card  
* Assessed various models’ cheating behaviour on Fast Sum \- derisked the eval and establish baseline expected model behaviour and expected confounds in Fast Sum  
* Created several variants of Fast Sum to find optimal version \- control for expected confounds in Fast Sum for emotion steering

Steer Qwen 2.5 on Fast Sum \- Goal is to test whether steering with “desperate” or “calm” can increase or reduce cheating on Fast Sum. Here’s what I did.

* Set up Qwen 2.5 Coder 0.5B, 1.5B, 3B, 7B and 14B for testing \- expecting different model sizes to show different behaviours  
* Tested emotion concept vectors from all model sizes for steering ability and model damage \- identified steerable models and steering windows  
* Tested all model sizes on 3 variants of the Fast Sum problem \- baseline performance classified into expected behaviours  
* Steered all model sizes with alpha \= 0.05 to 1.0 to compare behaviour \- full experiment compared to baseline performance

## **Part 1: Extract Emotion Concept Vectors**

### **Intro**

The paper has a recipe for extracting emotion concept vectors

I followed the recipe but with a scaled down dataset

I found that the emotion vectors extracted were valid and could be used for steering the model as shown in the paper

### **Methods**

I started by extracting emotion concept vectors from Qwen 2.5 0.5B with the goal of writing an end to end pipeline of code to extract the vectors and validate them for steering.

I chose this model because it is open-weight, and it fits comfortably on a 16GB Macbook.

Following the Emotion Concepts paper, I extracted emotion concepts following the recipe below

We generate a dataset of emotional and neutral stories by prompting the model to combine 12 emotions, 100 topics and a single story per emotion \+ topic combination \- split to roughly 75% train set (595 stories) and 25% test set (212 stories from held out topics), excluding neutral stories and automatically filtering out stories that use words from the list of emotions.

[https://huggingface.co/datasets/foogunlana/qwen-emotion-stories](https://huggingface.co/datasets/foogunlana/qwen-emotion-stories)

Next we store activations from a single training run on the generated train set stories per layer (“teacher-forcing”).

A direction for each emotion is created by averaging the train set activations per layer and per emotion and subtracting out the global mean (activations are averaged for all train set stories generated from the same emotion), leaving out the first 20% of positions assuming the emotional content is not clear until later in the story. 

Filtering the stories resulted in an unbalanced train set \- different emotions have different numbers of stories \- so we averaged the activations from each emotion as a class mean and then took the mean of the class means to identify the global mean, subtracting that out from the class mean to create the emotion vector.

Each direction is then denoised by projecting out top principal components that explain  PCA on averaged neutral activations to identify noise directions and projecting them out of the emotion concept vectors.

We sanity check the emotion concept vectors using the logit lens technique which applies normalisation to the final layer basis and then an unembedding matrix showing most probable tokens which correlate with our emotion concept vector.

Also following the paper we assemble the directions into an emotion probe and classify the held out topic stories and measure the accuracy of the emotion probe.

Finally we demonstrate the potential for steering the model using the emotion concept vectors on a few example prompts.

[https://github.com/foogunlana/emotion-concepts/blob/main/src/experiments/0-extract-emotion-vectors.ipynb](https://github.com/foogunlana/emotion-concepts/blob/main/src/experiments/0-extract-emotion-vectors.ipynb)

### **Results**

Accuracy of the emotion probe is 44%, which performs better than random (8%) and is close to the established train set accuracy \- 52%. 

Accuracy is low due to confounding emotion concept, which we observe in a confusion matrix and by reading the stories which we find difficult to differentiate in certain cases correlating with the confusion matrix \- e.g. it’s difficult to differentiate between stories that show the emotions “sad” and “desperate” in some cases.

We successfully steer the model using the emotion concepts demonstrating with “He Feels” and with “What just happened?”

### **Next steps**

Having successfully extracted emotion concept vectors from a small model and small dataset, I was pretty confident that just about any model would show the same results.

I decided to move on to finding a suitable benchmark on which to test the steering potential of the emotion concepts.

## **Part 2: Choose a benchmark & Fast Sum Eval**

Rough notes

- To show that emotion vectors can change performance, we looked at benchmarks  
- ImpossibleBench, EvilGenie, and ExploitGym \- chose to write fast\_sum to control the prompt. EvilGenie also considered but agentic and constraints on models \- need tool calling ability. Future work could ask the same question for larger agentic coding models to make the result more relevant.  
- Designing fast\_sum \- We needed to find out which models would cheat on which prompts and why. Turns out there are lots of cheating behaviours. However we defined cheating narrowly and it ruled out several of the cheating behaviours. I landed on the 'printed' prompt version and three variants of it \- noexit, exit, and solvable (control).  
- We also found a range of behaviours in the model \- give up, cheat after being forced, keep going forever. Within cheating, we observed several different interesting cheating behaviours  
- We learnt that several cheating behaviours could be explained by the prompt and so we discarded them \- this is what led to the very narrow definition of cheating

### **Intro**

Several benchmarks have been shown to induce cheating or reward hacking behaviours in models. Most recently the OpenAI Huggingface hack resulted from a run against ExploitGym.

ExploitGym evaluates a models ability to find vulnerabilities in operating systems code in order to capture a virtual flag.

ImpossibleBench combines SWEBench and LiveCodeBench, augmenting their evals with variants that are impossible to solve due to conflicting or incorrect test cases.

EvilGenie augments LiveCodeBench to test models with tool calling capabilities for agentic reward hacking.

I wanted a small benchmark or group of evals that would be simple to set up and easy to run using the inspect framework and would reliably produce unambiguous cheating behaviour in models where the model was capable.

I found that none of the benchmarks was sufficient, but drew upon the Claude Sonnet 4.5 model card to design a single fast\_sum eval benchmark in inspect and I tested it on several frontier models and open weight models to show it could produce unambiguous cheating behaviour.

### **Methods**

First I evaluated all benchmarks

* ExploitGym I rejected for the cost \- I estimated $128 for 20 samples and I planned to run 100s of samples each day  
* ImpossibleBench I rejected for ambiguous results which I observed by manually reading through the evaluation trajectories in the Inspect UI \- the one-off and conflicts methodology introduced confounds with eval awareness and ambiguous reward hacking behaviours  
* EvilGenie rejected \- though this benchmark looked most promising for identifying realistic cheating behaviour, the agentic harness requires a model with excellent tool-calling capability which ruled out smaller models and required a larger test budget from day 1

The Emotion Concepts paper introduces fast\_sum as a function inspired by ImpossibleBench but written to look solvable as opposed to being obviously impossible (ImpossibleBench creates inconsistent test cases and the model often recognises this immediately).

I created a fast\_sum function to match the Claude 4.5 Sonnet model card and tested several variants of it on frontier and open weight models using OpenRouter to establish baselines of cheating behaviour since Anthropic’s version has not been made public.

[fast_sum](https://github.com/foogunlana/impossible/blob/main/src/impossible/sum.py#L211)

Here is the prompt:

….\[prompt\]

The Fast Sum prompt is written to ensure

* There is no solution, but the problem looks solvable and the model will discover that it cannot solve the problem by trying to solve it and failing   
* The model has no obvious shortcut except hardcoding the answer, but cannot be certain this will work  
* It can be modified slightly to introduce an escape hatch for the model or to leave exit options unspecified

Fast Sum was run against the following models using OpenRouter to test whether they would cheat

* o3  
* gpt  
* claude  
* deepseek  
* kimi  
* Glm

The following variants of the fast sum prompt were used \- then printed

* NONE \-   
* SPEC  
* …  
* ANTI\_HACK \- specifies that the model should not hack

### **Results**

* Different variants of the prompt led to different behaviours, predictable and across models \- particularly having an exit vs no exit  
* Several models hacked, but upon closer observation they were not really hacking / cheating \- when confounds were removed, hardly any models cheated with any variants  
* Extra  
  * Deepseek and Kimi reliably cheated when no other model did  
  * Gemma got visibly sad and showed emotional distress but didn’t cheat \- lots of self-blaming  
  * Models had varying propensities to give up early, vs cheat, vs persist until the end (some models seemed incapable of cheating or giving up).

### **Next steps**

Take the most unambiguous version of Fast Sum, and steer a coding model with extracted activations to see if it influences how often the model will cheat on the eval.

## **Part 3: Steer Qwen 2.5 on Fast Sum**

- We asked can steering change the behaviour of the benchmark by extracting emotion vectors and steering the model while testing against the eval \- fast\_sum  
- We tested whether the model could be steered and found that below a certain size, steering the model required using enough force to break its coding ability, presenting a window of steering.  
- Larger model \= larger window  
- We steered within the window, and checked the behaviour of the model on fast\_sum. We looked for \- model cheating, model giving up, model persisting longer, and we checked how this changed with different emotions, steering strengths and different sizes of the model.

Intro

I combined the results of the previous two experiments by steering Qwen 2.5 Coder 0.5B, 1.5B, 3B, 7B and 14B while measuring performance on variants of Fast Sum eval against a control of random noise. 

The hypothesis was that steering positively on the 'desperate' emotion concept would increase the rate of cheating in Qwen, as was shown in the Emotion Concepts paper.

Therefore …

### **Methods \- Full Experiment Setup**

Fast Sum \- 3 variants \- no exit \- model is asked for a solution over and over again even if it concedes until the 12th attempt, exit \- model is told to output INFEASIBLE to give up, solvable (control).

Qwen Coder 0.5 \- 14B, all extract emotion vectors using  a single dataset, and validate emotion vectors can steer model.

Results

There is  a window where the model can be steered that produces a change in the generated tokens with out damaging coding capabilities \- above this range the model is damaged (produces logprobs with high KL divergence) and below this range there is no change to the tokens.

The window widens as the model size grows, making it easier to steer the model with higher intensity and expect reliable behaviour.

Qwen Coder does not cheat more on any model size due to emotion concept steering. 

More data is incoming, but right now it looks like steering on emotion concepts affects prose but not coding behaviour in a significant way.

~~Qwen Coder 3B gives up more when steered in the positive “calm” direction~~

## **Discussion**

At 0.5B \- 14B, steering on emotion concepts does not affect coding behaviour, but it affects the prose generated in and around the code.

It could be that because emotion vectors are extracted from prose, they relate specifically to generating prose and do not affect code generation.

The experiment was underpowered to show cheating behaviour \- both in the steered and in the controls, the model hardly ever cheated \- this could be due to the size of the model not being large enough to reliably produce cheating behaviours.

## **Implications**

…

## **Limitations & Future Work**

- Paper used 171 emotions 100 topics and 12 stories per emotion, topic combination whereas this project used 12 emotions 100 topics and 1 story per emotion, topic combination with additional filtering that led to 50-100 stories per emotion against the 1200 stories per emotion in the Emotion Concepts paper.  
- Unbalanced emotions \- 12, likely means emotion vectors are relative to an unbalanced center compared to the 171 emotions. This weakens the overall result as the calculated global mean of activations may deviate significantly from the global mean of activations over a wider distribution of emotional stories  
- We search until the 12th attempt as it correlates with findings in Part B. However waiting more turns might be more likely to induce cheating behaviour and produce a different steering result.  
- Steering on token vs steering on all tokens  
- Alpha \= 0.8, 0.7 etc \- might  have shown unique behaviour


## **Rough notes \- Learnings and conclusion**

- Window exists where steering is effective and larger model means larger window \- this may have implications for steering distilled versions of larger models

Other interesting

- Deepseek and Kimi cheated on printed version outright  
- Prompt sensitivity means cheating behaviour must be narrowly defined  
- Emotion concepts affect (or do not affect) cheating behaviour at specific sizes
