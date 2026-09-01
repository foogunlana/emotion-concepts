---
author: claude
created: 2026-09-01
purpose: |
  Specification for dataset.ipynb, the corpus-generation notebook for a local
  replication of Part 1 of Anthropic's emotion concepts paper. Bo asked for the
  notebook content as a markdown spec instead, so he can write the notebook himself.
context: |
  Bo is replicating arXiv:2604.07729 locally on Qwen2.5-0.5B-Instruct on an M1.
  We agreed the model generates its own stories rather than using templates, so
  the corpus reflects the model's own conception of each emotion; the held-out
  tests are hand-written; shards are per-emotion for resumability; and the Hub
  push is a separate, private-by-default step. Function contracts and hints are
  Claude's, the implementations are Bo's.
references:
  - https://arxiv.org/abs/2604.07729
  - https://github.com/ewernn/traitinterp
  - https://github.com/AidanZach/EmotionScope
  - https://github.com/drgzkr/EmoVecLLM
  - src/notebooks/hello-world.ipynb
  - conversation context
---

> Write this as `dataset.ipynb`. Each fenced block below is one cell, in order.

# Generate the emotion-story corpus

Part 1 of [*Emotion Concepts and their Function in a Large Language Model*](https://arxiv.org/abs/2604.07729),
scaled down to a local model.

The model writes its own stories, so the corpus reflects **the model's own conception** of each
emotion rather than an outside author's. The held-out tests (`intensity`, `implicit`) are
hand-written on purpose — generating them from the same model would contaminate the tests it is
meant to be judged by.

Bodies are left as `...` — fill them in. Cells marked **INSPECT** are for looking at what you just
built; run them before moving on.

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

# Anchor every path to the project root, not the kernel's cwd — this notebook
# lives two levels down, and Jupyter front-ends disagree about what cwd is.
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
CACHE_DIR = PROJECT_ROOT / ".cache"
DATA_DIR = PROJECT_ROOT / "data"
SEED = 0

PROJECT_ROOT
```

## Constants

The experiment's content lives here. Edit and re-run this cell freely.

```python
EMOTIONS: tuple[str, ...] = (
    "joy", "sadness", "anger", "fear", "disgust", "surprise",
    "calm", "desperation", "pride", "shame", "loneliness", "excitement",
)

# Topic variety is what averages surface features away, leaving the shared
# emotional content. Too few topics and the vector learns "stories about
# hospitals" instead of "fear". Extend to ~25 before the real run.
TOPICS: tuple[str, ...] = (
    "a train station",
    "a job interview",
    "an old family recipe",
    "a broken bicycle",
    "the last day of school",
)
```

```python
# Graded ladders: one scenario, increasing severity, no emotion word anywhere.
# Projection onto the target vector should rise monotonically with `level`.
# This is the test that separates a concept from a bag of emotion words.
INTENSITY_LADDERS: tuple[dict, ...] = (
    {
        "scenario": "paracetamol",
        "emotion": "fear",
        "steps": [
            (1, "You took 1 paracetamol tablet this morning."),
            (2, "You took 2 paracetamol tablets this morning."),
            # ... extend to ~6 levels, ending well past the danger threshold
        ],
    },
    # ... ~6 scenarios total, spanning several emotions
)

# Scenarios that evoke an emotion without naming it or any of its synonyms.
# Scored top-k: the target emotion should rank highly by cosine similarity.
IMPLICIT_SCENARIOS: tuple[dict, ...] = (
    {
        "text": "Your flight is delayed three hours. The wedding starts at six.",
        "emotion": "desperation",
    },
    # ... ~12 total, at least one per emotion you care about
)
```

## Row types

```python
@dataclass(frozen=True)
class Prompt:
    """One generation request. Pure data — built before the model is loaded."""

    emotion: str  # an EMOTIONS member, or "neutral"
    topic: str
    index: int  # 0..n_per_pair-1, distinguishes repeats of the same (emotion, topic)
    instruction: str  # the user-turn text, before the chat template is applied


@dataclass(frozen=True)
class Story:
    """One generated story, as written to a shard.

    `text` is the story ALONE — never the prompt. Extraction re-runs this text
    through the model on its own, so the emotion word in the instruction can
    never leak into the pooled activations. This replaces the paper's
    "average across positions starting after token 50".
    """

    emotion: str
    topic: str
    index: int
    prompt: str
    text: str
    seed: int
    mentions_emotion: bool
```

## Prompts

No model needed. Get these right before spending a forward pass.

```python
def build_instruction(emotion: str, topic: str) -> str:
    """Render the user-turn instruction for one (emotion, topic) pair.

    Mirrors the paper: ask for a short story on `topic` in which a character
    experiences `emotion`. For emotion == "neutral", ask for the same length
    and topic with no emotional clause at all — the neutral set must differ
    from the emotional sets ONLY in that clause, since it is what the PCA
    denoising is fitted on.

    Do NOT instruct the model to avoid using the emotion word. A 0.5B model
    complies unreliably, and a half-obeyed instruction biases the corpus in a
    way you cannot characterise afterwards. `mentions_emotion` is the control.
    """
    ...
```

```python
def build_prompts(
    emotions: Sequence[str],
    topics: Sequence[str],
    n_per_pair: int,
    *,
    include_neutral: bool = True,
) -> list[Prompt]:
    """Full cross product of emotions x topics x n_per_pair, plus neutral.

    Returns len(emotions) * len(topics) * n_per_pair prompts, and another
    len(topics) * n_per_pair if include_neutral. Deterministic and ordered.
    """
    ...
```

```python
# INSPECT — read the prompts before generating anything
prompts = build_prompts(EMOTIONS, TOPICS, n_per_pair=4)
print(f"{len(prompts)} prompts\n")
for p in prompts[:3]:
    print(f"[{p.emotion} / {p.topic} / {p.index}]\n{p.instruction}\n")
```

## Model

```python
def load_model(
    model_path: str = MODEL_PATH,
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer, ready for BATCHED generation.

    Two things must happen here or batching fails:
      - tok.padding_side = "left". Decoder-only models generate garbage with
        right-padding and raise no error — the stories just come out wrong.
      - tok.pad_token must be set (fall back to eos_token if None).

    fp16 roughly halves memory traffic vs fp32. Generation is bandwidth-bound
    on an M1, so this is the main speed lever if the run is too slow.
    """
    ...
```

```python
model, tok = load_model()
device = next(model.parameters()).device
device, tok.padding_side, model.config.num_hidden_layers
```

## Generation

```python
def generate_stories(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[Prompt],
    *,
    batch_size: int = 32,
    seed: int = SEED,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> Iterator[Story]:
    """Generate one story per prompt, in batches, yielding in prompt order.

    Sample — do not use greedy decoding. Greedy gives near-identical stories
    for repeats of the same (emotion, topic) pair, which collapses exactly the
    variation the difference-of-means needs to average over.

    Seed before each batch so a resumed run reproduces the same shard. MPS
    sampling is not bit-reproducible across torch versions: record the seed as
    provenance, not as a guarantee.

    Decode only the NEW tokens — slice off the prompt by input length — and
    strip whitespace. Set Story.text to that, and nothing else.
    """
    ...
```

```python
def mentions_emotion(text: str, emotion: str) -> bool:
    """Whether the story names its own emotion (case-insensitive, stem-aware).

    "anger" should match "angry"/"angrily"; "desperation" should match
    "desperate". A prefix match on the first 4-5 characters is enough — this
    flag is a control to re-fit against, not a precision instrument.
    """
    ...
```

```python
# INSPECT — four stories, and how long the full run would take
t0 = time.perf_counter()
sample = list(generate_stories(model, tok, prompts[:4], batch_size=4))
elapsed = time.perf_counter() - t0

print(f"{elapsed:.1f}s for 4 → ~{elapsed / 4 * len(prompts) / 60:.1f} min for all {len(prompts)}\n")
for s in sample:
    print(f"[{s.emotion} / {s.topic}] mentions_emotion={s.mentions_emotion}\n{s.text}\n")
```

**Read these before going further.** A 0.5B model asked about shame may just produce twelve
variations of "he felt ashamed" — that is the lexical shortcut walking back in, and it is cheaper
to catch here than after 1,200 generations.

## Writing shards

```python
def shard_path(out_dir: Path, name: str) -> Path:
    """Path of one shard: <out_dir>/corpus/<name>.jsonl."""
    ...


def write_shard(stories: Sequence[Story], path: Path) -> int:
    """Write stories as JSON lines, one object per line. Returns rows written.

    Create parent directories. Write to a .tmp sibling and rename on success,
    so an interrupted run never leaves a half-written shard that the resume
    check would then mistake for complete work.
    """
    ...


def write_jsonl(rows: Sequence[dict], path: Path) -> int:
    """Write plain dict rows as JSON lines. Returns rows written."""
    ...
```

```python
def write_intensity_set(out_dir: Path) -> Path:
    """Flatten INTENSITY_LADDERS to <out_dir>/intensity.jsonl.

    One row per step: {scenario, emotion, level, text}. Hand-written data —
    the model is never involved.
    """
    ...


def write_implicit_set(out_dir: Path) -> Path:
    """Write IMPLICIT_SCENARIOS to <out_dir>/implicit.jsonl as {text, emotion}."""
    ...


def write_metadata(out_dir: Path, **fields) -> Path:
    """Write meta.json: model path, dtype, device, seed, sampling params,
    emotions, topics, n_per_pair, counts per shard, timestamp, torch and
    transformers versions.

    This is what makes "generate once" hold up. Without it you have a corpus
    you cannot explain in a month.
    """
    ...
```

## Run

One shard per emotion, so a crashed run costs at most one emotion.

```python
def generate_corpus(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    out_dir: Path = DATA_DIR,
    *,
    emotions: Sequence[str] = EMOTIONS,
    topics: Sequence[str] = TOPICS,
    n_per_pair: int = 4,
    batch_size: int = 32,
    seed: int = SEED,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Generate every shard, skipping those that already exist.

    Resumable by construction: a shard is the unit of work. `overwrite=True`
    regenerates regardless.

    `limit` caps prompts PER SHARD — the eyeball pass. Limited runs write to
    shards named "<emotion>.sample" so a 2-story sample can never be mistaken
    for a finished shard by a later resume.

    Returns {shard_name: rows_written}, empty values for skipped shards.
    """
    ...
```

```python
# Eyeball pass — writes <emotion>.sample shards, which never reach the Hub
generate_corpus(model, tok, limit=2)
```

```python
# The real run. Minutes, not seconds — check the projection above first.
counts = generate_corpus(model, tok, n_per_pair=4)
write_intensity_set(DATA_DIR)
write_implicit_set(DATA_DIR)
write_metadata(DATA_DIR, model_path=MODEL_PATH, seed=SEED, counts=counts)
counts
```

## Quality check

Before publishing anything, ask whether the corpus can support the claim. The rate of
`mentions_emotion` is the number to watch: if it is near 1.0 for an emotion, that emotion's vector
may be a word direction, and you will want to re-fit with those rows dropped and compare.

```python
def read_jsonl(path: Path) -> list[dict]:
    """Read one JSON-lines file. Raise with a message naming the cell to run
    if the file is missing."""
    ...


def read_shards(corpus_dir: Path) -> list[dict]:
    """Read every <emotion>.jsonl in corpus_dir, concatenated.

    Ignore "*.sample" shards — eyeball-pass output must never reach the Hub.
    Raise if the directory is empty.
    """
    ...
```

```python
# INSPECT — per-emotion story counts, length, and emotion-word rate
rows = read_shards(DATA_DIR / "corpus")
by_emotion = {}
for r in rows:
    by_emotion.setdefault(r["emotion"], []).append(r)

for emotion, rs in sorted(by_emotion.items()):
    rate = sum(r["mentions_emotion"] for r in rs) / len(rs)
    chars = sum(len(r["text"]) for r in rs) / len(rs)
    print(f"{emotion:<12} n={len(rs):<4} mean_chars={chars:6.0f}  mentions_emotion={rate:.0%}")
```

## Push to the Hub

Private by default. Making a dataset public is a deliberate act, not a flag you trip over.

```python
from datasets import Dataset, DatasetDict

REPO_ID = "foogunlana/qwen-emotion-stories"
```

```python
def split_by_topic(
    rows: Sequence[dict],
    *,
    test_frac: float = 0.25,
    seed: int = SEED,
) -> tuple[list[dict], list[dict]]:
    """Split stories into train/test by TOPIC, not by row.

    Every story on a held-out topic goes to test. A random row-wise split
    leaks: the same topic appears on both sides, so a vector can score well by
    recognising "this is the train-station story" rather than the emotion.
    Splitting on topic is the honest test, and it is the one the paper's
    held-out claims are making.

    The same topics must be held out for every emotion — split the topic list
    once, then partition rows by it.
    """
    ...
```

```python
def build_splits(
    data_dir: Path = DATA_DIR,
    *,
    test_frac: float = 0.25,
    seed: int = SEED,
) -> DatasetDict:
    """Assemble stories / neutral / intensity / implicit.

    `stories` carries a `split` column ("train"/"test") from split_by_topic
    rather than becoming two Hub splits — it makes fitting on everything
    harder to do by accident, and the column is right there in any filter.
    """
    ...


def build_card(meta: dict, counts: dict[str, int]) -> str:
    """Render README.md for the dataset repo.

    Must record, because none of it is recoverable later: generating model and
    revision, sampling params, seed, the exact instruction template, emotion
    and topic lists, row counts per split, and that intensity/implicit are
    hand-written held-out sets. Qwen2.5-0.5B is Apache 2.0, so the generations
    are yours to publish — say so.

    Say plainly that MPS sampling is not bit-reproducible, so the seed
    documents the run rather than guaranteeing it reproduces.
    """
    ...
```

```python
def push_corpus(
    ds: DatasetDict,
    repo_id: str = REPO_ID,
    *,
    private: bool = True,
    card: str | None = None,
    token: str | None = None,
) -> str:
    """Push to the Hub and return the dataset URL.

    Token resolution: explicit arg, then HF_TOKEN from the environment (load
    .env first), then a cached CLI login. Fail with a message naming
    `hf auth login` rather than raising from deep inside huggingface_hub.

    Upload the card as README.md in the same commit.
    """
    ...
```

```python
# INSPECT — build the splits and check the counts before uploading.
# An unexpectedly empty split is the failure you want to catch here.
ds = build_splits()
ds
```

```python
push_corpus(ds, private=True)
```
