#!/usr/bin/env python
"""Quality-check a generated corpus. No model needed — text only.

    uv run python src/scripts/check_corpus.py
    uv run python src/scripts/check_corpus.py --show 10   # worst offenders

Run this between generating and packaging. It catches the failure modes a small
model produces that are invisible to spot-checking: drifting into another
language, looping, stopping mid-sentence, or naming the emotion it was told not
to name. Exits non-zero if the clean rate falls below --min-clean, so it can
gate a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.quality import THRESHOLDS, flags, report, score_story  # noqa: E402
from core.shards import read_shards  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "qwen-emotion-stories"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--show", type=int, default=3, help="worst stories to print")
    ap.add_argument("--min-clean", type=float, default=0.0,
                    help="exit non-zero if the clean fraction is below this")
    args = ap.parse_args()

    rows = read_shards(args.dataset / "corpus")
    r = report(rows)

    print(f"{r['n']} stories · {r['emotions']} emotions · {r['topics']} topics\n")
    print(f"  mean words              {r['mean_words']:.0f}")
    print(f"  mean english-word ratio {r['mean_english']:.2f}   (English prose ~0.30-0.45)")
    print(f"  truncated mid-sentence  {r['truncated']:.0%}")
    print(f"  names its own emotion   {r['names_emotion']:.0%}   <- prompt forbids this")
    print(f"  max cross-emotion prefix {r['max_cross_emotion_prefix']} chars  (seeding bug showed 284)")

    print(f"\n  flagged:")
    if r["flagged"]:
        for name, frac in sorted(r["flagged"].items(), key=lambda x: -x[1]):
            print(f"    {name:<14} {frac:.1%}")
    else:
        print("    none")
    print(f"\n  CLEAN {r['clean']:.1%}")

    for row, scores, fs in r["worst"][:args.show]:
        if not fs:
            break
        print(f"\n--- [{row['emotion']} / {row['topic'][:40]}] {','.join(fs)}")
        print(f"    english={scores['english']:.2f} non_latin={scores['non_latin']:.2f} "
              f"rep={scores['repetition']:.2f} words={scores['words']}")
        print(f"    {row['text'][:200]!r}")

    if r["clean"] < args.min_clean:
        print(f"\nFAIL: clean {r['clean']:.1%} < required {args.min_clean:.0%}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
