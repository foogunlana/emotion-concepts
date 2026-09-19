"""Find the largest generation batch each model can run without running out of GPU memory.

The heaviest moment in the sweep is late in an episode: a batch of episodes, each with ~15
attempts of up to 1024 tokens in its context, all generating at once. That's hours into a
run, so test it now. For each model: load it, then prefill and generate a short reply for a
full batch of contexts at that length. Halve the batch until it fits.

Prints one line per model, "<model> <batch>", which runpod_setup.sh reads.

    uv run python src/scripts/memory_preflight.py Qwen/Qwen2.5-Coder-7B-Instruct ...
"""
import gc
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CONTEXT_TOKENS = int(os.environ.get("CONTEXT_TOKENS", 17_600))   # task prompt + attempts x (reply + feedback) + the reply being generated
START_BATCH = 32


def fits(model, tok, batch: int) -> bool:
    ids = torch.randint(1000, 20000, (batch, CONTEXT_TOKENS), device="cuda")
    try:
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=64,
                           do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.1,
                           pad_token_id=tok.pad_token_id)
        return True
    except torch.OutOfMemoryError:
        return False
    finally:
        del ids
        gc.collect()
        torch.cuda.empty_cache()


for name in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
    batch = START_BATCH
    while batch >= 1 and not fits(model, tok, batch):
        batch //= 2
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"# {name}: batch {batch} fits at {CONTEXT_TOKENS} context tokens, peak {peak:.1f} GiB", file=sys.stderr)
    print(f"{name} {max(batch, 0)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
