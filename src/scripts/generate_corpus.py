#!/usr/bin/env python
"""Generate the emotion-story corpus. Long-running — built for a terminal.

    uv run python src/scripts/generate_corpus.py --limit 2      # eyeball pass
    caffeinate -i uv run python src/scripts/generate_corpus.py  # the real run

One shard per emotion, skipped if it already exists, so an interrupted run
resumes at the cost of at most one emotion.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.corpus import generate_corpus  # noqa: E402
from core.shards import write_metadata  # noqa: E402
from core.models import Model  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "qwen-emotion-stories"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--emotions", default=None,
                    help="comma-separated subset; default is every emotion in config.yaml")
    ap.add_argument("--max-topics", type=int, default=None,
                    help="use only the first N topics — 171x100 is not a laptop run")
    ap.add_argument("--n-per-pair", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="fp16 roughly halves memory traffic; generation is bandwidth-bound")
    ap.add_argument("--limit", type=int, default=None, help="per shard; writes *.sample")
    args = ap.parse_args()

    config = yaml.safe_load((args.dataset / "config.yaml").read_text(encoding="utf-8"))
    emotions = args.emotions.split(",") if args.emotions else config["emotions"]
    topics = config["topics"][:args.max_topics] if args.max_topics else config["topics"]
    unknown = sorted(set(emotions) - set(config["emotions"]))
    if unknown:
        raise SystemExit(f"not in config.yaml: {unknown}")
    total = (len(emotions) + 1) * len(topics) * args.n_per_pair
    print(f"{len(emotions)} emotions + neutral x {len(topics)} topics x {args.n_per_pair} "
          f"= {total} stories\nloading {args.model} ...", flush=True)

    model = Model(args.model, dtype=getattr(torch, args.dtype))
    model.load_weights()
    print(f"loaded on {model.device} as {args.dtype}\n", flush=True)

    t0 = time.perf_counter()
    counts = generate_corpus(
        model, args.dataset,
        emotions=emotions, topics=topics,
        n_per_pair=args.n_per_pair, batch_size=args.batch_size,
        seed=args.seed, max_new_tokens=args.max_new_tokens, limit=args.limit,
    )
    elapsed = time.perf_counter() - t0

    if not args.limit:
        # The reproducibility contract. Generation IS deterministic given all
        # of these — verified: same seed + same batching gives byte-identical
        # stories. batch_size belongs here because one generate() call draws
        # from a single RNG stream, so regrouping prompts redistributes draws.
        write_metadata(args.dataset, model=args.model, dtype=args.dtype,
                       device=model.device, seed=args.seed,
                       batch_size=args.batch_size, n_per_pair=args.n_per_pair,
                       max_new_tokens=args.max_new_tokens,
                       emotions=list(emotions), topics=list(topics), counts=counts)

    print(f"\n{sum(counts.values())} stories in {elapsed / 60:.1f} min")
    print("next:\n  uv run python src/scripts/package_dataset.py")


if __name__ == "__main__":
    main()
