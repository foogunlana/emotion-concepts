"""Reading and writing the corpus on disk.

Deliberately small: one atomic writer, two readers, and provenance. Anything
that decides *what* to write — shard names, resume policy, how the graded
ladders flatten — lives in the notebook, where you can see it and change it.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import transformers


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    """Write rows as JSON lines, atomically. Returns rows written.

    Writes to a .tmp sibling and renames on success, so an interrupted run
    leaves either a complete file or nothing — never a partial one that a
    resume check ("does this file exist") would mistake for finished work.

    Consumes lazily, so a generator of stories streams to disk rather than
    being materialised first. ensure_ascii=False keeps the model's typographic
    quotes and em-dashes readable in the file instead of escaped to \\uXXXX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} — run generate_corpus first")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_shards(corpus_dir: Path) -> list[dict]:
    """Every finished shard, concatenated.

    Skips *.sample shards — those are eyeball-pass output and must never reach
    the corpus or the Hub.
    """
    paths = sorted(p for p in corpus_dir.glob("*.jsonl") if not p.stem.endswith(".sample"))
    if not paths:
        raise FileNotFoundError(f"no shards in {corpus_dir}")
    return [row for p in paths for row in read_jsonl(p)]


def write_metadata(out_dir: Path, **fields) -> Path:
    """Provenance for the run, with library versions captured automatically.

    Pass batch_size as well as seed: generation seeds per batch, so batch size
    is part of what determines the output.
    """
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        **fields,
    }
    path = out_dir / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path
