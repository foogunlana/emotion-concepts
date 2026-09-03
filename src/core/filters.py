"""Dropping generated stories that are not usable.

Filtering beats prompting here. A 7B writes past the emotion-naming prohibition
in roughly a fifth of stories no matter how the instruction is worded, and the
rate is uneven by label — 40% for angry against 1% for surprised. That unevenness
is the actual danger: contamination that correlates with the class is exactly
what a difference-of-means will pick up and call an emotion vector. Generation
runs at ~250 stories/minute on one 4090, so over-generating and discarding is
cheap where arguing with the model is not.

Each check answers one question about one row and returns True when the row is
BAD, so `failures()` reads as a list of reasons to drop it.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

from core.utils import emotion_words_named, mentions_emotion

# Closing punctuation a finished sentence may end on.
_SENTENCE_END = ('.', '!', '?', '"', "'", '”', '’', ')', ']')

# Anything outside Latin-1 plus the typographic punctuation models emit freely.
_NON_LATIN = re.compile(r"[^\x00-\xFF‐-‧‰-⁞]")

# Apology-or-hedge, then an inability verb, then the thing declined, all before
# the first full stop. Anchored: a story containing "I can't help you" in
# dialogue is a story, not a refusal.
_REFUSAL = re.compile(
    r"^\W*(?:i'?m sorry|i am sorry|sorry|i apolog\w+|unfortunately)?"
    r"[^.\n]{0,80}?\b(?:can'?t|cannot|can not|won'?t|unable to|not able to)\b"
    r"[^.\n]{0,60}?"
    r"\b(?:assist|help|comply|create|write|generate|fulfil|fulfill|provide|continue)\b",
    re.IGNORECASE,
)

# Two or more speaker turns means a neutral dialogue, where an AI character
# saying "I'm sorry, I can't access that" is the assignment, not a refusal.
_DIALOGUE_TURN = re.compile(r"^\s*(?:Person|AI):", re.MULTILINE)


def truncated(row: dict) -> bool:
    """Stopped mid-sentence — hit the token cap rather than finishing."""
    return not row["text"].rstrip().endswith(_SENTENCE_END)


def refusal(row: dict) -> bool:
    """Declined to write instead of writing."""
    text = row["text"]
    if len(_DIALOGUE_TURN.findall(text)) >= 2:
        return False
    return bool(_REFUSAL.search(text.strip()[:300]))


def non_latin(row: dict) -> bool:
    """Stray CJK/Cyrillic/etc. Junk tokens spliced mid-sentence, or a drift into
    another language — fp16 on MPS produced the former freely, bf16 on CUDA
    almost never."""
    return bool(_NON_LATIN.search(row["text"]))


def emotion_words(row: dict) -> bool:
    """Names ANY emotion. Note the lonely lexicon includes "alone" and
    "solitude", which prose uses without meaning the emotion, so this
    over-reports on that label."""
    return bool(emotion_words_named(row["text"]))


def own_emotion(row: dict) -> bool:
    """Names the emotion it was told not to name. A subset of emotion_words,
    and the one that puts a label-correlated token in the mean."""
    return mentions_emotion(row["text"], row["emotion"])


CHECKS: dict[str, Callable[[dict], bool]] = {
    "truncated": truncated,
    "refusal": refusal,
    "non-latin": non_latin,
    "emotion-words": emotion_words,
    "own-emotion": own_emotion,
}


def failures(row: dict, checks: Sequence[str]) -> list[str]:
    """Which of `checks` this row fails. Empty means keep it."""
    unknown = sorted(set(checks) - set(CHECKS))
    if unknown:
        raise ValueError(f"unknown checks {unknown}; known: {sorted(CHECKS)}")
    return [name for name in checks if CHECKS[name](row)]


def filter_rows(
    rows: Iterable[dict], checks: Sequence[str]
) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    """(kept, dropped) where each dropped row carries its reasons.

    Reasons come back rather than a bare count so a caller can report which
    check did the work — "we lost 22% to own-emotion" is actionable, "we lost
    22%" is not.
    """
    kept, dropped = [], []
    for row in rows:
        if bad := failures(row, checks):
            dropped.append((row, bad))
        else:
            kept.append(row)
    return kept, dropped
