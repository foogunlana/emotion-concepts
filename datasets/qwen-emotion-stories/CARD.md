# Qwen2.5-0.5B emotion stories

Corpus for a local replication of Part 1 of [*Emotion Concepts and their Function
in a Large Language Model*](https://arxiv.org/abs/2604.07729).

<!-- STATS -->

## Held-out tests

`intensity` and `implicit` are written by claude Opus-5 (though they ideally should be hand-written, but it's okay that they're not written by the model being tested)

## Fields

`emotion`, `topic`, `index`, `prompt`, `text`, `batch_seed`, `mentions_emotion`, `split`.

A story is identified by `(emotion, topic, index)`. `batch_seed` is per batch. One batch is generated from one RNG seed so it can be reproduced.

`text` is the story generated.

`mentions_emotion` flags stories that name their own emotion.

## Reproducibility

Generation is deterministic: the same seed with the same batch composition produces identical stories based on the unique combination of: model, dtype, device, torch version, seed, batch size, and prompt order.

`config.yaml` in this repo is the full definition: emotions, topics, and both
held-out test sets.
