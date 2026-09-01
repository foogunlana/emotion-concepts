"""Generate the emotion-story corpus used to fit emotion vectors.

Replication of Part 1 of "Emotion Concepts and their Function in a Large
Language Model" (arXiv:2604.07729), scaled down to a local model.

The model writes its own stories, so the corpus reflects the model's own
conception of each emotion rather than an outside author's. Held-out test
sets (intensity, implicit) are hand-written on purpose: generating them from
the same model would contaminate the tests it is meant to be judged by.

Outputs, all under --out:

    corpus/<emotion>.jsonl   one shard per emotion, Story rows
    corpus/neutral.jsonl     topic-matched, emotionally flat, for PCA denoising
    intensity.jsonl          hand-written graded ladders
    implicit.jsonl           hand-written scenarios that never name the emotion
    meta.json                provenance for the whole run

Run `push_dataset.py` afterwards to publish. This script never touches the Hub.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# --- experiment constants ----------------------------------------------------

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


# --- prompts (no model required) ---------------------------------------------


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


def build_prompts(
    emotions: Sequence[str],
    topics: Sequence[str],
    n_per_pair: int,
    *,
    include_neutral: bool = True,
) -> list[Prompt]:
    """Full cross product of emotions × topics × n_per_pair, plus neutral.

    Returns len(emotions) * len(topics) * n_per_pair prompts, and another
    len(topics) * n_per_pair if include_neutral. Deterministic and ordered —
    print it and read it before spending a forward pass.
    """
    ...


# --- generation --------------------------------------------------------------


def load_model(
    model_path: str,
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


def generate_stories(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: Sequence[Prompt],
    *,
    batch_size: int = 32,
    seed: int = 0,
    max_new_tokens: int = 120,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> Iterator[Story]:
    """Generate one story per prompt, in batches, yielding in prompt order.

    Sample — do not use greedy decoding. Greedy gives near-identical stories
    for repeats of the same (emotion, topic) pair, which collapses exactly the
    variation the difference-of-means needs to average over.

    Seed before each batch so a resumed run reproduces the same shard. Note
    that MPS sampling is not bit-reproducible across torch versions: record the
    seed as provenance, not as a guarantee.

    Decode only the newly generated tokens — slice off the prompt by input
    length — and strip whitespace. Set Story.text to that, and nothing else.
    """
    ...


def mentions_emotion(text: str, emotion: str) -> bool:
    """Whether the story names its own emotion (case-insensitive, stem-aware).

    "anger" should match "angry"/"angrily"; "desperation" should match
    "desperate". A prefix match on the first 4-5 characters is enough here —
    this flag is a control to re-fit against, not a precision instrument.
    """
    ...


# --- io ----------------------------------------------------------------------


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


# --- orchestration -----------------------------------------------------------


def generate_corpus(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    out_dir: Path,
    *,
    emotions: Sequence[str] = EMOTIONS,
    topics: Sequence[str] = TOPICS,
    n_per_pair: int = 4,
    batch_size: int = 32,
    seed: int = 0,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Generate every shard, skipping those that already exist.

    Resumable by construction: a shard is the unit of work, so a crashed run
    costs at most one emotion. `overwrite=True` regenerates regardless.

    `limit` caps prompts PER SHARD — `--limit 2` is the eyeball pass. Limited
    runs write to shards named "<emotion>.sample" so a 2-story sample can
    never be mistaken for a finished shard by a later resume.

    Returns {shard_name: rows_written}, empty values for skipped shards.
    """
    ...


def main() -> None:
    """CLI.

    --model        path or hub id (default: the local Qwen2.5-0.5B-Instruct)
    --out          data directory (default: data/)
    --emotions     subset to generate, defaults to all
    --n-per-pair   stories per (emotion, topic) pair
    --batch-size   generation batch
    --limit        cap prompts per shard, for the eyeball pass
    --seed
    --overwrite
    --print        echo generated stories to stdout as they land

    Time the first batch and print a projected total before committing to the
    full run — you want to know inside a minute whether the budget holds.
    """
    ...


if __name__ == "__main__":
    main()
