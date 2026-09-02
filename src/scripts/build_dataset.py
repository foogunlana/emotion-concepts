#!/usr/bin/env python
"""Build the local dataset folder so it can be synced to the Hub as-is.

    uv run python src/scripts/build_dataset.py
    hf upload foogunlana/qwen-emotion-stories datasets/qwen-emotion-stories \
        --repo-type=dataset --delete "*"

The folder IS the Hub repo: what you see locally is exactly what gets published.
Nothing here talks to the network — uploading is a separate, deliberate step.

    datasets/<name>/
        config.yaml        INPUT  — hand-edited, defines everything
        corpus/*.jsonl     stories, one shard per emotion (split column added here)
        intensity.jsonl    derived from config.yaml
        implicit.jsonl     derived from config.yaml
        README.md          generated: frontmatter + docs/dataset-card.md

The README frontmatter maps each config to its data files. Without it the Hub
cannot tell the three subsets apart and load_dataset fails with a confusing
schema mismatch. They are configs rather than splits because their schemas
differ, which a DatasetDict does not allow.

This script rewrites the corpus shards in place to add the `split` column. It is
idempotent — rerun it freely, including after regenerating a single shard.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.shards import read_jsonl, write_jsonl  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "qwen-emotion-stories"
DEFAULT_CARD = PROJECT_ROOT / "docs" / "dataset-card.md"
DEFAULT_REPO_ID = "foogunlana/qwen-emotion-stories"

FRONTMATTER = """---
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

"""


def assign_splits(dataset_dir: Path, *, test_frac: float, seed: int) -> dict[str, int]:
    """Add a `split` column to every corpus shard, in place.

    Split by TOPIC, not by row: every story on a held-out topic goes to test,
    and the same topics are held out for every emotion. A row-wise split leaks.

    Topics are drawn from the whole corpus, not per shard, so the partition is
    identical across emotions.
    """
    shards = sorted(p for p in (dataset_dir / "corpus").glob("*.jsonl")
                    if not p.stem.endswith(".sample"))
    if not shards:
        raise SystemExit(f"no shards in {dataset_dir / 'corpus'} — generate the corpus first")

    topics = sorted({r["topic"] for p in shards for r in read_jsonl(p)})
    rng = random.Random(seed)
    rng.shuffle(topics)
    held_out = set(topics[: max(1, round(len(topics) * test_frac))])

    counts = {}
    for path in shards:
        rows = [{**r, "split": "test" if r["topic"] in held_out else "train"}
                for r in read_jsonl(path)]
        counts[path.stem] = write_jsonl(rows, path)
    return counts


def write_test_sets(dataset_dir: Path, config: dict) -> tuple[int, int]:
    """Flatten the hand-written held-out sets out of config.yaml."""
    n_intensity = write_jsonl(
        (
            {"scenario": l["scenario"], "emotion": l["emotion"],
             "level": s["level"], "text": s["text"]}
            for l in config["intensity"]
            for s in l["steps"]
        ),
        dataset_dir / "intensity.jsonl",
    )
    n_implicit = write_jsonl(
        (dict(s) for s in config["implicit"]),
        dataset_dir / "implicit.jsonl",
    )
    return n_intensity, n_implicit


def render_stats(rows: list[dict], n_intensity: int, n_implicit: int, model: str,
                 *, test_frac: float, seed: int) -> str:
    """The generated block. Prose is hand-written in docs/dataset-card.md; only
    the numbers live here, so they can never go stale."""
    n_train = sum(r["split"] == "train" for r in rows)
    topics = {r["topic"] for r in rows}

    banner = "" if len(topics) >= 10 else (
        f"> **Preliminary.** {len(rows)} stories across {len(topics)} topic(s) — a\n"
        f"> plumbing run, not the corpus. Difference-of-means needs many topics to\n"
        f"> average surface features away, so vectors fitted on this would be topic\n"
        f"> vectors as much as emotion vectors. To be regenerated.\n\n"
    )

    return f"""{banner}## Configs

| config | rows | source |
|---|---|---|
| `stories` | {len(rows)} ({n_train} train / {len(rows) - n_train} test) | model-generated |
| `intensity` | {n_intensity} | hand-written, held out |
| `implicit` | {n_implicit} | hand-written, held out |

{len(set(r["emotion"] for r in rows))} emotions over {len(topics)} topics, generated with `{model}`.
Split assigned with `test_frac={test_frac}`, `seed={seed}`.

```python
from datasets import load_dataset
stories = load_dataset("{DEFAULT_REPO_ID}", "stories")["train"]
```"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--card", type=Path, default=DEFAULT_CARD, help="hand-written prose")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="recorded in the card")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = ap.parse_args()

    config = yaml.safe_load((args.dataset / "config.yaml").read_text(encoding="utf-8"))

    counts = assign_splits(args.dataset, test_frac=args.test_frac, seed=args.seed)
    n_intensity, n_implicit = write_test_sets(args.dataset, config)

    rows = [r for p in sorted((args.dataset / "corpus").glob("*.jsonl"))
            if not p.stem.endswith(".sample") for r in read_jsonl(p)]

    body = args.card.read_text(encoding="utf-8")
    stats = render_stats(rows, n_intensity, n_implicit, args.model,
                         test_frac=args.test_frac, seed=args.seed)
    if "<!-- STATS -->" not in body:
        raise SystemExit(f"{args.card} has no <!-- STATS --> marker")
    (args.dataset / "README.md").write_text(
        FRONTMATTER + body.replace("<!-- STATS -->", stats), encoding="utf-8")

    for name, n in sorted(counts.items()):
        print(f"  corpus/{name}.jsonl  {n:>4} rows")
    print(f"  intensity.jsonl     {n_intensity:>4} rows")
    print(f"  implicit.jsonl      {n_implicit:>4} rows")
    print(f"  README.md           written")

    train_topics = {r["topic"] for r in rows if r["split"] == "train"}
    test_topics = {r["topic"] for r in rows if r["split"] == "test"}
    print(f"\ntopics: {len(train_topics)} train / {len(test_topics)} test")
    if not train_topics:
        print("WARNING: train split is empty — too few topics to split")

    rel = args.dataset.relative_to(PROJECT_ROOT)
    print(f"\nto publish:\n  hf upload {args.repo_id} {rel} --repo-type=dataset --delete \"*\"")


if __name__ == "__main__":
    main()
