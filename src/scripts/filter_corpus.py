#!/usr/bin/env python
"""Drop unusable stories from a generated corpus. No model needed — text only.

    uv run python src/scripts/filter_corpus.py --dry-run          # what would go
    uv run python src/scripts/filter_corpus.py                    # rewrite shards
    uv run python src/scripts/filter_corpus.py --drop own-emotion,non-latin
    uv run python src/scripts/filter_corpus.py --balance          # equal-sized shards

Run between generating and packaging. Shards are rewritten in place, and the
corpus shrinks unevenly — emotions whose vocabulary is hard to write around lose
more rows than others (lonely kept 41 of 100 where neutral kept 98).

--balance then trims every shard to the smallest survivor, so each emotion's
mean is computed over the same number of stories. Without it, vector quality
varies by emotion in step with how hard that emotion's word was to avoid, which
is a confound dressed as a result. The cost is throwing away good stories, so
over-generating with a larger --n-per-pair first is the better order.

Checks (default: all of them):
    truncated      stopped mid-sentence at the token cap
    refusal        declined to write instead of writing
    non-latin      stray CJK/Cyrillic junk tokens
    emotion-words  names any emotion at all
    own-emotion    names the emotion it was told not to name
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.filters import CHECKS, filter_rows  # noqa: E402
from core.shards import read_jsonl, write_jsonl  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "qwen-emotion-stories"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--drop", default=",".join(CHECKS),
                    help=f"comma-separated subset of: {','.join(CHECKS)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="report only — do not rewrite the shards")
    ap.add_argument("--balance", action="store_true",
                    help="trim every shard to the smallest one after filtering")
    ap.add_argument("--cap", type=int, default=None,
                    help="explicit rows per shard; implies --balance")
    ap.add_argument("--seed", type=int, default=0, help="which rows --balance keeps")
    ap.add_argument("--show", type=int, default=0, help="print this many dropped stories")
    args = ap.parse_args()

    checks = [c.strip() for c in args.drop.split(",") if c.strip()]
    unknown = sorted(set(checks) - set(CHECKS))
    if unknown:
        raise SystemExit(f"unknown checks {unknown}; known: {sorted(CHECKS)}")

    paths = sorted(p for p in (args.dataset / "corpus").glob("*.jsonl")
                   if not p.stem.endswith(".sample"))
    if not paths:
        raise SystemExit(f"no shards in {args.dataset / 'corpus'}")

    reasons: collections.Counter = collections.Counter()
    examples: list[tuple[dict, list[str]]] = []
    survivors: dict[Path, list[dict]] = {}
    total_before = 0

    print(f"dropping: {', '.join(checks)}\n")
    for path in paths:
        rows = read_jsonl(path)
        kept, dropped = filter_rows(rows, checks)
        survivors[path] = kept
        total_before += len(rows)
        for _, bad in dropped:
            reasons.update(bad)
        examples.extend(dropped[:2])

        lost = len(rows) - len(kept)
        bar = "" if not lost else "  " + ", ".join(
            f"{n} {name}" for name, n in collections.Counter(
                b for _, bad in dropped for b in bad).most_common())
        print(f"  {path.stem:<12} {len(kept):>4}/{len(rows):<4} kept  (-{lost:>3}){bar}")

    # Balance AFTER filtering: the cap has to be the smallest SURVIVING shard,
    # not the smallest generated one.
    cap = args.cap or (min(len(k) for k in survivors.values()) if args.balance else None)
    if cap:
        rng = random.Random(args.seed)
        print(f"\n  balancing to {cap} rows per shard:")
        for path, kept in survivors.items():
            if len(kept) > cap:
                # Sample, then restore file order — a shard that is shuffled on
                # disk makes every later diff unreadable for no benefit.
                keep = set(rng.sample(range(len(kept)), cap))
                survivors[path] = [r for i, r in enumerate(kept) if i in keep]
            print(f"    {path.stem:<12} {len(survivors[path]):>4}"
                  f"{'' if len(kept) <= cap else f'  (-{len(kept) - cap})'}")

    total_after = sum(len(k) for k in survivors.values())
    if not args.dry_run:
        for path, kept in survivors.items():
            write_jsonl(kept, path)

    lost = total_before - total_after
    print(f"\n  {total_after}/{total_before} kept, {lost} dropped ({lost / max(1, total_before):.1%})")
    print(f"  by check: {dict(reasons.most_common())}")
    print("\n  (dry run — nothing written)" if args.dry_run else "\n  shards rewritten")

    for row, bad in examples[:args.show]:
        print(f"\n--- [{row['emotion']}] {','.join(bad)}\n    {row['text'][:200]!r}")


if __name__ == "__main__":
    main()
