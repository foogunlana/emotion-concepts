"""Extract and cache activations. The only step that needs a GPU.

Run on the pod, copy the .pt back, do every other step locally:

    python src/scripts/extract_activations.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --batch-size 8 \
        --out data/acts/qwen2.5-7b.pt

Backgrounded, since it outlives an SSH connection:

    nohup python -u src/scripts/extract_activations.py ... > extract.log 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.acts import extract, save_acts
from core.models import Model
from core.shards import read_shards

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--corpus", type=Path,
                    default=PROJECT_ROOT / "datasets" / "qwen-emotion-stories" / "corpus")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "acts" / "acts.pt")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="hidden_states holds every layer at full sequence length; "
                         "this is the knob that OOMs first on a big model")
    ap.add_argument("--dtype", choices=DTYPES, default="float32",
                    help="bfloat16 on a big model if fp32 will not fit; the pooled "
                         "output is cast to fp32 either way")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows — smoke-test the whole path before committing "
                         "a GPU-hour to it")
    args = ap.parse_args()

    rows = read_shards(args.corpus)
    if args.limit:
        rows = rows[: args.limit]

    model = Model(args.model, dtype=DTYPES[args.dtype])
    model.load_weights()
    print(f"{args.model} on {model.device}, {len(rows)} rows, batch {args.batch_size}")

    X = extract(model, [r["text"] for r in rows], batch_size=args.batch_size)

    path = save_acts(
        X, args.out,
        model=args.model,
        dtype=args.dtype,
        n_rows=len(rows),
        n_layers=X.shape[1] - 1,
        hidden=X.shape[2],
        # Row order is not recoverable from the tensor, and every mask
        # downstream is positional. Carry enough to rebuild it.
        row_ids=[{"emotion": r["emotion"], "topic": r["topic"],
                  "index": r["index"], "split": r["split"]} for r in rows],
    )

    mb = path.stat().st_size / 1e6
    print(f"\n{X.shape} -> {path} ({mb:.0f} MB)")

    if mb > 1000:
        # Single SSH transfers die at a fixed offset around 1.7GB.
        print(
            f"\nOver 1GB — split before pulling:\n"
            f"  split -b 300m {path} {path}.part_\n"
            f"then pull each part with `ssh pod \"cat <part>\" > <part>`, "
            f"and `cat {path.name}.part_* > {path.name}`."
        )


if __name__ == "__main__":
    main()
