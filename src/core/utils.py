"""Text helpers — not about models, not about storage."""

from __future__ import annotations

import re


EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "joyful": ("joy", "joys", "joyful", "joyfully", "joyous", "overjoyed",
            "happy", "happily", "happiness", "delight", "delighted",
            "elated", "elation", "glad", "cheerful"),
    "sad": ("sad", "sadly", "sadness", "sorrow", "sorrowful", "grief",
                "grieving", "unhappy", "miserable", "misery", "heartbroken",
                "melancholy", "mournful"),
    "angry": ("anger", "angry", "angrily", "angered", "furious", "fury",
              "furiously", "enraged", "rage", "irate", "livid", "indignant"),
    "afraid": ("fear", "fearful", "fearfully", "afraid", "scared", "frightened",
             "frightening", "terrified", "terror", "panic", "panicked",
             "dread", "alarmed"),
    "disgusted": ("disgust", "disgusted", "disgusting", "revolted", "revulsion",
                "repulsed", "repulsive", "nauseated", "sickened", "loathing"),
    "surprised": ("surprise", "surprised", "surprising", "astonished",
                 "astonishment", "amazed", "amazement", "startled", "stunned",
                 "shocked"),
    "calm": ("calm", "calmly", "calmness", "serene", "serenity", "tranquil",
             "peaceful", "relaxed", "composed"),
    "desperate": ("desperate", "desperately", "desperation", "despair",
                    "despairing", "hopeless", "hopelessness", "frantic",
                    "frantically"),
    "proud": ("pride", "proud", "proudly", "prideful"),
    "ashamed": ("shame", "ashamed", "shameful", "shamefully", "humiliated",
              "humiliation", "embarrassed", "embarrassment", "mortified"),
    "lonely": ("lonely", "loneliness", "lonesome", "alone", "isolated",
                   "isolation", "solitary", "solitude"),
    "excited": ("excited", "excitement", "exciting", "excitedly", "thrilled",
                   "thrill", "exhilarated", "eager", "enthusiastic"),
}

_PATTERNS: dict[str, re.Pattern] = {
    emotion: re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    for emotion, words in EMOTION_WORDS.items()
}


def _pattern(emotion: str) -> re.Pattern:
    """Curated word list where we have one, else the bare label.

    The paper uses 171 emotion labels; hand-curating synonym lists for all of
    them is not worth it. The fallback matches only the label itself, so it
    under-reports rather than inventing false positives.
    """
    if emotion in _PATTERNS:
        return _PATTERNS[emotion]
    return re.compile(r"\b" + re.escape(emotion) + r"\b", re.IGNORECASE)


def emotion_words_named(text: str) -> dict[str, int]:
    """Every emotion in the lexicon this text names, with hit counts.

    `mentions_emotion` asks one question about the label a story was given.
    This asks the same question of all twelve at once, which is what catches a
    proud story that spends two sentences on shame — no rule broken by the
    letter of the old prohibition, but a foreign concept in the passage the
    vector is pooled from.

    Note the lonely lexicon includes "alone" and "solitude", which ordinary
    prose uses without meaning the emotion. Expect this to over-report lonely
    slightly; it is a screening tool, not a verdict.
    """
    return {
        emotion: len(hits)
        for emotion, pattern in _PATTERNS.items()
        if (hits := pattern.findall(text))
    }


def mentions_emotion(text: str, emotion: str) -> bool:
    """Whether `text` explicitly names `emotion` or a close synonym.
    """
    if emotion == "neutral":
        return False
    return bool(_pattern(emotion).search(text))


# A sentence ends on . ! ? possibly followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r'[.!?]["\')\]”’]*(?=\s|$)')


def trim_to_sentence(text: str, min_keep: float = 0.5) -> str:
    """Drop a dangling half-sentence left by the token cap.

    A generation that hits max_new_tokens ends mid-clause. That tail is not
    neutral for extraction: it is the one part of every story with no completed
    thought in it, and it is present in almost every row of a capped run.

    Returns the text unchanged when trimming would cost more than `min_keep` of
    it — a story with no sentence boundary at all is a different failure, and
    silently deleting most of it would hide that rather than report it.
    """
    text = text.rstrip()
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if not ends or ends[-1] == len(text):
        return text
    if ends[-1] < len(text) * min_keep:
        return text
    return text[:ends[-1]]
