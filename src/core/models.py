from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


import torch
from pathlib import Path
from core.types import Prompt

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
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def load_weights(self):
        if self.model is not None:
            return
        self.tok = AutoTokenizer.from_pretrained(self.modelname, cache_dir=CACHE_DIR)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname, cache_dir=CACHE_DIR, dtype=self.dtype).to(self.device).eval()

    def gen(self, batch: list[Prompt], max_new_tokens=512, full_transcript: bool = False, sample=False, system: str | None = None):
        """`system` prepends a system turn to every prompt in the batch.

        Without one a small instruct model refuses a share of the dark premises
        outright — see core.prompts.SYSTEM for why that share is not noise.
        """
        self.load_weights()

        prefix = [{"role": "system", "content": system}] if system else []
        texts = [
            self.tok.apply_chat_template(prefix + [{"role": "user", "content": p.instruction}], tokenize=False, add_generation_prompt=True)
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
