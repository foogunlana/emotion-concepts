"""Publish the generated corpus to the Hugging Face Hub.

Reads whatever `generate_corpus.py` left in the data directory and pushes it
as a single dataset with four splits. Never generates anything — if a shard is
missing, that is an error to report, not a gap to fill.

Splits:

    stories    model-generated, one row per story, with `topic` retained
    neutral    topic-matched flat passages, for fitting the PCA denoising
    intensity  hand-written graded ladders (held out)
    implicit   hand-written unnamed-emotion scenarios (held out)

Private by default. Making a dataset public is a deliberate act, not a flag
you trip over.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from datasets import Dataset, DatasetDict

DEFAULT_REPO_ID = "foogunlana/qwen-emotion-stories"


def read_jsonl(path: Path) -> list[dict]:
    """Read one JSON-lines file. Raises FileNotFoundError with a message that
    names the script to run if the file is missing."""
    ...


def read_shards(corpus_dir: Path) -> list[dict]:
    """Read every <emotion>.jsonl in corpus_dir, concatenated.

    Ignore "*.sample" shards — those are eyeball-pass output and must never
    reach the Hub. Raise if the directory is empty.
    """
    ...


def split_by_topic(
    rows: Sequence[dict],
    *,
    test_frac: float = 0.25,
    seed: int = 0,
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


def build_splits(
    data_dir: Path,
    *,
    test_frac: float = 0.25,
    seed: int = 0,
) -> DatasetDict:
    """Assemble the four splits from the data directory.

    `stories` carries a `split` column ("train"/"test") from split_by_topic
    rather than becoming two separate Hub splits — keeping it one split makes
    it harder to accidentally fit on everything, and the column is right there
    in any downstream filter.
    """
    ...


def build_card(meta: dict, counts: dict[str, int]) -> str:
    """Render README.md for the dataset repo.

    Must record, because none of it is recoverable later: generating model and
    revision, sampling params, seed, the exact instruction template, emotion
    and topic lists, row counts per split, and that intensity/implicit are
    hand-written held-out sets. State the licence position too — Qwen2.5-0.5B
    is Apache 2.0, so the generations are yours to publish.

    Say plainly that MPS sampling is not bit-reproducible, so the seed
    documents the run rather than guaranteeing it reproduces.
    """
    ...


def push_corpus(
    ds: DatasetDict,
    repo_id: str,
    *,
    private: bool = True,
    card: str | None = None,
    token: str | None = None,
) -> str:
    """Push to the Hub and return the dataset URL.

    Token resolution order: explicit arg, then HF_TOKEN from the environment
    (load .env first), then a cached CLI login. Fail with a clear message
    naming `hf auth login` rather than letting huggingface_hub raise from
    somewhere deep.

    Upload the card as README.md in the same commit.
    """
    ...


def main() -> None:
    """CLI.

    --data       data directory written by generate_corpus.py (default: data/)
    --repo-id    default: foogunlana/qwen-emotion-stories
    --public     opt in to a public dataset; private otherwise
    --test-frac  fraction of TOPICS held out
    --seed
    --dry-run    build the splits and print the card and row counts, push nothing

    Print the row count per split before pushing. A split that is unexpectedly
    empty is the failure you want to catch here, not after upload.
    """
    ...


if __name__ == "__main__":
    main()
