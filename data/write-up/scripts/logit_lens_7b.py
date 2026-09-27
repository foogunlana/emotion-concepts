"""Logit lens of the 12 emotion vectors, Qwen2.5-Coder-7B-Instruct (Appendix A). Run from repo root:
    uv run python data/write-up/scripts/logit_lens_7b.py [layer]    # default 19, the steering layer

Same method as logit_lens in src/core/acts.py: final RMSNorm, then the unembedding, top-k tokens.
Downloads only what the lens needs: the lm_head shard (~1.1 GB), and model.norm.weight (7 KB) read with an
HTTP range request from its 4.3 GB shard, instead of downloading that shard.
"""
import json, struct, sys
import requests
import torch
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

REPO, K = "Qwen/Qwen2.5-Coder-7B-Instruct", 8
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 19
CACHE = ".cache"

d = torch.load("data/steer-runpod/qwen2.5-coder-7b-instruct/vectors.pt", map_location="cpu")
assert d["meta"]["model"] == REPO and d["layers"][LAYER] == LAYER
emotions, V = d["emotions"], d["V"][:, LAYER].float()          # (12, 3584)

wmap = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json", cache_dir=CACHE)))["weight_map"]

def read_tensor_range(name):
    """One tensor out of a remote safetensors shard, via byte ranges (header, then just that tensor)."""
    url = hf_hub_url(REPO, wmap[name])
    get = lambda a, b: requests.get(url, headers={"Range": f"bytes={a}-{b}"}, allow_redirects=True).content
    n = struct.unpack("<Q", get(0, 7))[0]
    meta = json.loads(get(8, 8 + n - 1))[name]
    assert meta["dtype"] == "BF16"
    a, b = meta["data_offsets"]
    raw = bytearray(get(8 + n + a, 8 + n + b - 1))
    return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(meta["shape"])

with safe_open(hf_hub_download(REPO, wmap["lm_head.weight"], cache_dir=CACHE), "pt", device="cpu") as f:
    W_U = f.get_tensor("lm_head.weight").float()                 # (vocab, hidden)
gain = read_tensor_range("model.norm.weight").float()            # (hidden,)
eps = json.load(open(hf_hub_download(REPO, "config.json", cache_dir=CACHE)))["rms_norm_eps"]
tok = AutoTokenizer.from_pretrained(REPO, cache_dir=CACHE)

# Qwen2RMSNorm, as model.norm(v) in acts.logit_lens
v = V * torch.rsqrt(V.pow(2).mean(-1, keepdim=True) + eps) * gain
top = (v @ W_U.T).topk(K, dim=-1).indices

BYTES = {c: b for b, c in bytes_to_unicode().items()}

def show(t):
    s = tok.decode([t])
    if "\ufffd" in s:                                            # part of a multi-byte character: show the bytes
        s = repr(bytes(BYTES[c] for c in tok.convert_ids_to_tokens(t)))
    elif not s.strip():
        s = repr(s)
    s = s.strip().replace("|", "\\|")
    return f"`` {s} ``" if "`" in s else f"`{s}`"

rows = ["| emotion | top tokens |", "|---|---|"]
for i, e in enumerate(emotions):
    rows.append(f"| {e} | " + " ".join(show(t) for t in top[i].tolist()) + " |")
print("\n".join(rows))   # pasted into data/write-up/logit_lens_7b.md, with glosses and caption added by hand
