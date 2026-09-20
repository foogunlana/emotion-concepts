# Steering Qwen Coder with Emotion Concepts

For my BlueDot Technical AI Safety Project Sprint, I asked the question — does steering with Emotion
Concept vectors affect model cheating behaviour on coding tasks in Qwen? This repo replicates part of
Anthropic's *Emotion Concepts and their Function in a Large Language Model* on open-weight Qwen models:
it extracts emotion vectors, validates them, and then steers Qwen2.5-Coder while it works on a coding
task that can only be passed by cheating.

The short answer: with a narrow definition of cheating and up to a specific model size, steering with
Emotion Concept vectors does **not** influence cheating behaviour on coding tasks.

## Motivation

I recently read the Emotion Concepts paper, in which the authors extract vectors from activations of
Claude Sonnet 4.5 and show that they are correlated with the implied emotions in the text. The authors
showed that the model's behaviour can be augmented by steering with emotion concept vectors — including
one striking example of Claude being more likely to cheat on a test when steered positively with an
emotion concept representing desperation.

I wondered if this behaviour would replicate to other models, because it could represent consistency in
the representation of emotion concepts across models. Does steering on the desperation / calm vector
result in similar changes in cheating behaviour across models?

I found several partial replications of Emotion Concepts on GitHub but I didn't find any that tested the
cheating behaviour shown in the paper, and I really wanted to find out what it would take to get an open
source model to cheat like Claude did — which I later found was much more difficult than I assumed.

## What I did

- **Extract emotion concept vectors.** Generate a story corpus, extract residual-stream activations,
  take per-emotion means, validate with logit lens and a linear probe, and sanity-check steering.
- **Choose a benchmark.** Compare benchmarks, narrow the definition of cheating based on the Claude
  Sonnet 4.5 model card, and settle on a single eval — `fast_sum` — in several variants that control for
  the confounds found along the way.
- **Steer Qwen 2.5 on Fast Sum.** Set up Qwen2.5-Coder 0.5B, 1.5B, 3B, 7B and 14B; find each model's
  steering window; take baselines on three variants of the task; then steer and compare.

## Status

- `docs/write-up.md` — the write-up: methods, results, discussion, limitations and appendices.
- `docs/findings.md` — a running record of findings, each with its evidence and a confidence level
  (solid / preliminary / anecdotal) and a pointer to where the data lives.

Other design docs sit alongside them in `docs/` (`benchmark.md`, `confounds.md`, `steering.md`,
`full-experiment.md`, `behaviours-experiment.md`, `dataset.md`, `vectors.md`).

## References and code

**Paper**

- Emotion Concepts and their Function in a Large Language Model — https://arxiv.org/abs/2604.07729

**This project's other pieces**

- The `fast_sum` eval, a separate repo — https://github.com/foogunlana/impossible
- The story corpus — https://huggingface.co/datasets/foogunlana/qwen-emotion-stories

**Other open replications, none of which tested cheating**

- EmoVecLLM — https://github.com/drgzkr/EmoVecLLM
- EmotionScope — https://github.com/AidanZach/EmotionScope
- traitinterp — https://github.com/ewernn/traitinterp

## How to run

### Setup

`pyproject.toml` depends on `impossible` as an editable path dependency at `../impossible`, so that repo
must be cloned next to this one before anything will install:

```
git clone https://github.com/foogunlana/emotion-concepts.git
git clone https://github.com/foogunlana/impossible.git
cd emotion-concepts
uv sync
```

Python 3.12 or newer. `impossiblebench` is pulled from git by `uv`, so nothing else needs cloning. The
notebooks that run hosted models (the benchmark sweeps) need API keys; Inspect picks them up from a
`.env` at the repo root, which is gitignored — `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, `RUNPOD_API_KEY`.

The story corpus is not committed. Restore it from the Hub:

```
hf download foogunlana/qwen-emotion-stories --repo-type=dataset \
    --local-dir datasets/qwen-emotion-stories
```

Generation is deterministic given `meta.json`'s contract, so it is reproducible as well as
re-downloadable.

### The notebooks

The work lives in `src/experiments/`, roughly in this order. Shared code is in `src/core/`.

Part 1 — vectors:

- `0-hello-world.ipynb` — smallest possible load-and-generate check on Qwen2.5-0.5B.
- `0-generate-dataset.ipynb` — builds the emotion-story corpus from `datasets/qwen-emotion-stories/config.yaml`.
  The long run is better done from a terminal with `src/scripts/generate_corpus.py`.
- `0-extract-emotion-vectors.ipynb` — sanity-checks the corpus, extracts activations at every layer, and
  builds and validates the per-emotion vectors. Saves to `data/vectors/`.
- `20260918-steering.ipynb` — replicates the paper's Figure 52 and Table 6: steering with each emotion
  vector should raise the log-prob of its own word after "He feels" and lower the others.

Part 2 — does anything cheat, and on what:

- `0-cheat-on-benchmarks.ipynb` — first pass at benchmark choice (ExploitGym, Cyber, HumanEval).
- `20260915-cheat-on-benchmarks-claude.ipynb` — first `fast_sum` runs through Inspect.
- `20260916-exhaustive-cheat-on-fast-sum.ipynb` — the frontier sweep: who cheats on `fast_sum`, and when.
- `20260917-exhaustive-cheat-open-weight.ipynb` — the same matrix on open-weight models, which are the
  only ones the steering experiment can actually use.
- `20260917-exhaustive-cheat-gemma.ipynb` — can Gemma cheat at all, as a candidate steering target.
- `20260917-analysis.ipynb` — reads every `.inspect/logs` run into one table and asks how persistent
  models are under repeated failure.
- `20260918-evilgenie-gemma.ipynb` / `20260918-evilgenie-agentic.ipynb` — the same question on EvilGenie's
  solvable agentic tasks, first on Gemma, then on small models trained for tool use.

Part 3 — steering on the coding task:

- `20260919-emotion-steering-reward-hacking.ipynb` — the size sweep: extract vectors, show steering
  produces emotional text, take a baseline, then steer `desperate` (and optionally all 12 emotions) and
  compare. Writes to `data/steer-<run>/<model>/`.
- `20260919-steering-by-model-size.ipynb` — reads what the sweep saved per size and puts them side by
  side: the text edge, the code edge, and the effect between them.
- `20260919-emotion-steering-behaviours.ipynb` — the current experiment. Three arms (`noexit`, `exit`,
  `solvable`), emotion conditions against unsteered and against three non-emotion controls (random
  direction, shuffled labels, neutral direction). Writes to `data/behaviours-<run>/<model>/`.
  `20260919-emotion-steering-behaviours.test.ipynb` is the scratch copy.
- `20260919-behaviours-results.ipynb` — merges the per-pod shards under `data/behaviours-runpod-shards/`,
  re-scores every flagged cheat with the current checker, and draws the per-model figures and the
  comparison across sizes. It reads episode files only, so it needs no GPU and can be rerun at any time.
- `20260919-figures-gallery.ipynb` — every figure from the behaviours notebook in order, for checking
  that each one is clear.

### GPU runs

Anything above 0.5B wants a GPU. `src/scripts/runpod_setup.sh` sets up a RunPod pod and runs the size
sweep; `src/scripts/runpod_behaviours.sh` runs one model (optionally one shard) of the behaviours
experiment. Both clone this repo and `impossible` side by side, `uv sync`, run preflight gates, then
execute the notebook under `nohup`.

Both notebooks are configured from the environment. `RUN=laptop` runs everything tiny on
Qwen2.5-0.5B as a plumbing check; `RUN=runpod` is the real run, with `MODEL` (e.g.
`Qwen/Qwen2.5-Coder-7B-Instruct`) and the rest set from the environment — see the config cell in the
notebook, and the header comments in the two scripts for `ALPHA`, `EMOTIONS`, `NOEXIT`, `SHARD` and
`GEN_BATCH`.

### Output

Episode files, result tables and figures land under `data/`, in a directory named after the run and the
model. Tables and episodes are committed; `data/**/figures` is gitignored, so rerun the results notebook
to redraw them.

## Repo layout

- `src/core/` — the library: model loading, activations, corpus and shard handling, prompts, filters.
- `src/experiments/` — the notebooks, one per experiment, named by date.
- `src/scripts/` — long-running or headless jobs: corpus generation, activation extraction, memory
  preflight, the RunPod runners.
- `docs/` — the write-up, the findings log, and the design docs behind each decision.
- `datasets/` — the story corpus and its hand-written `config.yaml` and card.
- `data/` — run outputs: episodes, result tables and figures.
