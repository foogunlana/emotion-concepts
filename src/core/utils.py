from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

import torch
from pathlib import Path

# project root cache
PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
CACHE_DIR = PROJECT_ROOT / ".cache"

class Model:
    def __init__(self, model = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.tok = None
        self.model = None
        self.modelname = model
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_weights(self):
        if self.model is not None:
            return
        self.tok = AutoTokenizer.from_pretrained(self.modelname, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname, cache_dir=CACHE_DIR, dtype=torch.float32).to(self.device).eval()

    def resp(self, msg: str, max_new_tokens=1024, full_transcript: bool = False):
        self.load_weights()

        msgs = [{"role": "user", "content": msg}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        if full_transcript:
            return self.tok.decode(out[0, :], skip_special_tokens=True)
        else:
            return self.tok.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
