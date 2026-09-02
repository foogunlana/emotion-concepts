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


def mentions_emotion(text: str, emotion: str) -> bool:
    """Whether `text` explicitly names `emotion` or a close synonym.
    """
    if emotion == "neutral":
        return False
    return bool(_pattern(emotion).search(text))
