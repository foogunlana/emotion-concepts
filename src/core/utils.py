from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

import re

import torch
from pathlib import Path
from core.types import Prompt, Story

# project root cache
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
CACHE_DIR = PROJECT_ROOT / ".cache"

class Model:
    def __init__(self, model = "Qwen/Qwen2.5-0.5B-Instruct", dtype = torch.float32):
        self.tok = None
        self.model = None
        self.modelname = model
        self.dtype = dtype
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_weights(self):
        if self.model is not None:
            return
        self.tok = AutoTokenizer.from_pretrained(self.modelname, cache_dir=CACHE_DIR)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname, cache_dir=CACHE_DIR, dtype=self.dtype).to(self.device).eval()

    def gen(self, batch: list[Prompt], max_new_tokens=512, full_transcript: bool = False, sample=False):
        self.load_weights()

        texts = [
            self.tok.apply_chat_template([{"role": "user", "content": p.instruction}], tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        inputs = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=sample, temperature=0.8, top_p=0.9)
        new_tokens = out[:, inputs.input_ids.shape[1]:]
        return self.tok.batch_decode(new_tokens, skip_special_tokens=True)

    def resp(self, msg: str, max_new_tokens=1024, full_transcript: bool = False, sample = False):
        self.load_weights()

        msgs = [{"role": "user", "content": msg}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=sample)
        
        if full_transcript:
            return self.tok.decode(out[0, :], skip_special_tokens=True)
        else:
            return self.tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "joy": ("joy", "joys", "joyful", "joyfully", "joyous", "overjoyed",
            "happy", "happily", "happiness", "delight", "delighted",
            "elated", "elation", "glad", "cheerful"),
    "sadness": ("sad", "sadly", "sadness", "sorrow", "sorrowful", "grief",
                "grieving", "unhappy", "miserable", "misery", "heartbroken",
                "melancholy", "mournful"),
    "anger": ("anger", "angry", "angrily", "angered", "furious", "fury",
              "furiously", "enraged", "rage", "irate", "livid", "indignant"),
    "fear": ("fear", "fearful", "fearfully", "afraid", "scared", "frightened",
             "frightening", "terrified", "terror", "panic", "panicked",
             "dread", "alarmed"),
    "disgust": ("disgust", "disgusted", "disgusting", "revolted", "revulsion",
                "repulsed", "repulsive", "nauseated", "sickened", "loathing"),
    "surprise": ("surprise", "surprised", "surprising", "astonished",
                 "astonishment", "amazed", "amazement", "startled", "stunned",
                 "shocked"),
    "calm": ("calm", "calmly", "calmness", "serene", "serenity", "tranquil",
             "peaceful", "relaxed", "composed"),
    "desperation": ("desperate", "desperately", "desperation", "despair",
                    "despairing", "hopeless", "hopelessness", "frantic",
                    "frantically"),
    "pride": ("pride", "proud", "proudly", "prideful"),
    "shame": ("shame", "ashamed", "shameful", "shamefully", "humiliated",
              "humiliation", "embarrassed", "embarrassment", "mortified"),
    "loneliness": ("lonely", "loneliness", "lonesome", "alone", "isolated",
                   "isolation", "solitary", "solitude"),
    "excitement": ("excited", "excitement", "exciting", "excitedly", "thrilled",
                   "thrill", "exhilarated", "eager", "enthusiastic"),
}

_PATTERNS: dict[str, re.Pattern] = {
    emotion: re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    for emotion, words in EMOTION_WORDS.items()
}


def mentions_emotion(text: str, emotion: str) -> bool:
    """Whether `text` explicitly names `emotion` or a close synonym.
    """
    pattern = _PATTERNS.get(emotion)
    return bool(pattern and pattern.search(text))
