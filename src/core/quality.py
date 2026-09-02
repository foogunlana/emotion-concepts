"""Corpus quality checks.

A small model produces failure modes that are invisible until they have already
polluted a vector: it drifts into another language, loops on a phrase, stops
mid-sentence, or names the emotion it was told not to name. Each check here
turns one of those into a number, so the corpus can be gated before extraction
rather than spot-checked by eye — which does not work when you cannot read the
language it drifted into.

Everything is dependency-free and runs on text alone. No model, no tokenizer.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

from core.utils import mentions_emotion

# Function words carry grammar, not topic, so their density is roughly constant
# across English prose (~30-45%) and collapses in any other language — including
# ones sharing the Latin alphabet, which a script check cannot see.
ENGLISH_FUNCTION_WORDS = frozenset("""
a an the and or but if while of to in on at by for with from into over under
is are was were be been being am do does did have has had will would can could
shall should may might must not no nor as that this these those there here
he she it they we you i him her them us his hers its their our your my me
what which who whom when where why how than then so because although though
""".split())

_WORD = re.compile(r"[a-zA-Z']+")
# Anything outside Latin-1 plus the typographic punctuation models emit freely.
_NON_LATIN = re.compile(r"[^\x00-\xFF‐-‧‰-⁞]")


def english_ratio(text: str) -> float:
    """Fraction of words that are English function words.

    English prose sits around 0.30-0.45. A story that drifts into Spanish or
    French keeps the Latin alphabet but loses these entirely, so this catches
    what a script check cannot. Short texts are noisy — treat < 20 words as
    unreliable.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return 0.0
    return sum(w in ENGLISH_FUNCTION_WORDS for w in words) / len(words)


def non_latin_ratio(text: str) -> float:
    """Fraction of characters outside Latin-1. Catches Arabic, CJK, Cyrillic."""
    return len(_NON_LATIN.findall(text)) / max(1, len(text))


def repetition(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are repeats — degenerate looping.

    Small models fall into "she walked to the door. she walked to the door."
    Clean prose sits near 0; a loop pushes it toward 1.
    """
    words = _WORD.findall(text.lower())
    if len(words) <= n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def truncated(text: str) -> bool:
    """Stopped mid-sentence — hit the token cap rather than finishing."""
    return not text.rstrip().endswith((".", "!", "?", '"', "'", "”"))


def score_story(row: dict) -> dict:
    """Per-story metrics. Cheap enough to run over the whole corpus."""
    text = row["text"]
    return {
        "words": len(text.split()),
        "english": english_ratio(text),
        "non_latin": non_latin_ratio(text),
        "repetition": repetition(text),
        "truncated": truncated(text),
        "names_emotion": mentions_emotion(text, row["emotion"]),
    }


# Thresholds are judgement calls, not laws. Tuned to flag the failure modes
# actually seen from Qwen2.5-0.5B rather than to be statistically principled.
THRESHOLDS = {
    "english": 0.20,      # below this it is probably not English
    "non_latin": 0.02,    # stray script fragments
    "repetition": 0.30,   # looping
    "words": 20,          # too short to carry an emotion
}


def flags(scores: dict) -> list[str]:
    """Which checks this story fails. Empty means clean."""
    out = []
    if scores["english"] < THRESHOLDS["english"]:
        out.append("not-english")
    if scores["non_latin"] > THRESHOLDS["non_latin"]:
        out.append("non-latin")
    if scores["repetition"] > THRESHOLDS["repetition"]:
        out.append("looping")
    if scores["words"] < THRESHOLDS["words"]:
        out.append("too-short")
    return out


def cross_emotion_overlap(rows: Sequence[dict]) -> int:
    """Longest shared prefix between stories of DIFFERENT emotions on the same
    topic. The seeding bug showed up here as 284 characters; independent
    generations sit under ~50. This is the check that catches a broken corpus
    outright rather than a merely mediocre one."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["topic"], r["index"]), []).append(r)
    worst = 0
    for group in groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i]["emotion"] == group[j]["emotion"]:
                    continue
                a, b = group[i]["text"], group[j]["text"]
                lcp = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y),
                           min(len(a), len(b)))
                worst = max(worst, lcp)
    return worst


def report(rows: Sequence[dict]) -> dict:
    """Corpus-level summary. `clean` is the fraction passing every check."""
    scored = [(r, score_story(r)) for r in rows]
    flagged = [(r, s, flags(s)) for r, s in scored]
    n = len(rows) or 1

    per_flag = Counter(f for _, _, fs in flagged for f in fs)
    return {
        "n": len(rows),
        "emotions": len({r["emotion"] for r in rows}),
        "topics": len({r["topic"] for r in rows}),
        "mean_words": sum(s["words"] for _, s in scored) / n,
        "mean_english": sum(s["english"] for _, s in scored) / n,
        "truncated": sum(s["truncated"] for _, s in scored) / n,
        "names_emotion": sum(s["names_emotion"] for _, s in scored) / n,
        "flagged": {k: v / n for k, v in per_flag.items()},
        "clean": sum(1 for _, _, fs in flagged if not fs) / n,
        "max_cross_emotion_prefix": cross_emotion_overlap(rows),
        "worst": sorted(flagged, key=lambda x: -len(x[2]))[:5],
    }
