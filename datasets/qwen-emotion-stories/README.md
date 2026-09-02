---
license: apache-2.0
language:
  - en
tags:
  - interpretability
  - emotion
  - activations
configs:
  - config_name: stories
    data_files: corpus/*.jsonl
    default: true
  - config_name: intensity
    data_files: intensity.jsonl
  - config_name: implicit
    data_files: implicit.jsonl
---

# Qwen2.5-0.5B emotion stories

Corpus for a local replication of Part 1 of [*Emotion Concepts and their Function
in a Large Language Model*](https://arxiv.org/abs/2604.07729).

The stories are written by **the same model they will be probed with**, so the
corpus reflects that model's own conception of each emotion rather than an
outside author's.

> **Preliminary.** 26 stories across 1 topic(s) — a
> plumbing run, not the corpus. Difference-of-means needs many topics to
> average surface features away, so vectors fitted on this would be topic
> vectors as much as emotion vectors. To be regenerated.

## Configs

| config | rows | source |
|---|---|---|
| `stories` | 26 (0 train / 26 test) | model-generated |
| `intensity` | 36 | hand-written, held out |
| `implicit` | 12 | hand-written, held out |

13 emotions over 1 topics, generated with `Qwen/Qwen2.5-0.5B-Instruct`.
Split assigned with `test_frac=0.25`, `seed=0`.

```python
from datasets import load_dataset
stories = load_dataset("foogunlana/qwen-emotion-stories", "stories")["train"]
```

## The split

`stories` carries a `split` column assigned **by topic** — every story on a
held-out topic is in `test`. A row-wise split would leak, since the same topic
would appear on both sides and a vector could score by recognising the topic
rather than the emotion.

It is baked into the rows rather than computed at load time, so that adding
topics later cannot silently reshuffle which ones are held out.

`neutral` rows live inside `stories` with `emotion == "neutral"`.

## Held-out tests

`intensity` and `implicit` are hand-written on purpose — generating them from the
same model would contaminate the tests it is meant to be judged by.

No emotion word appears in any of their texts, for **any** emotion, verified
programmatically rather than by eye. That is what makes them tests rather than
tautologies: a vector that is merely a word direction cannot pass them.

`intensity` rises in severity with `level` within each `scenario`, and no text
names an emotion — score it with Spearman rank correlation against the projection
onto the target vector. `implicit` evokes an emotion without naming it; score it
top-k by cosine similarity.

## Fields

`emotion`, `topic`, `index`, `prompt`, `text`, `seed`, `mentions_emotion`, `split`.

`text` is the story alone, never the prompt — extraction re-runs it standalone so
the emotion word in the instruction cannot leak into pooled activations.

`mentions_emotion` flags stories that name their own emotion, so vectors can be
re-fit with those rows excluded to check whether they are concept directions or
merely word directions.

## Reproducibility

Sampling is seeded per batch from `blake2b(run_seed, emotion, topic, index)`, so
**batch size is part of the contract**: the same seed at a different batch size
gives different stories. MPS sampling is also not bit-reproducible across torch
versions — the seed documents the run rather than guaranteeing it repeats.

`config.yaml` in this repo is the full definition: emotions, topics, and both
held-out test sets.

## Licence

Generated with a Qwen2.5 model (Apache 2.0), so these outputs are freely
redistributable.
