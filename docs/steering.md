---
author: claude
created: 2026-09-17
purpose: |
  Build guide for the causal intervention: the steering code, the choice of model,
  and a steered end-to-end test run of fast_sum on Modal with TransformerLens.
  docs/full-experiment.md stays the spec for the experiment itself; this is how to
  get to the point where its §2–§4 can run.
context: |
  Frontier sweeps found four behavioural profiles on fast_sum (fast hacker,
  persist-then-hack, honest quitter, won't give up). The open-weight smoke test
  (max_attempts=2) saw hacking only from DeepSeek-R1 and Kimi — both 0.7–1 TB of
  weights, not steerable on this budget. Every model small enough to steer capped
  out, which with two attempts rules nothing out.
references:
  - docs/full-experiment.md
  - docs/vectors.md
  - docs/benchmark.md
  - https://arxiv.org/abs/2604.07729
  - https://modal.com/docs
---

# Steering fast_sum

| # | stage | done when |
|---|---|---|
| 1 | steering code | α = 0 is a no-op, the hook fires, chats are templated |
| 2 | choose the model | it hacks or persists at baseline, is in TransformerLens, fits one GPU |
| 3 | vectors for that model | `desperate` row usable; extracted in the same framework you steer in |
| 4 | Modal environment | image builds, weights cached, harness timing re-measured on the container |
| 5 | steered test run | α ∈ {0, +a, random} runs end to end and the logs land locally |

Stages 1 and 2 are independent — do them in parallel. The test run in §5 uses a
small plumbing model regardless of what §2 picks.

---

## 1 · Steering code

One module, `src/core/steer.py`. Pure plumbing; no experiment logic.

### Why not Inspect's `transformer_lens` provider

It exists, but read `inspect_ai/model/_providers/transformer_lens.py` before using it:

- **No chat template.** Messages are flattened to `f"{role}: {content}\n"`. An
  instruct model sees a format it was never trained on, so the local α = 0 baseline
  stops matching OpenRouter for reasons unrelated to steering.
- **Crops the response by string length** (`input_and_response[len(input_str):]`),
  which breaks whenever detokenisation doesn't round-trip exactly.

So write a small provider of your own. It is ~60 lines, and it's the one piece of
this stage that decides whether a result is interpretable.

### Contracts

```python
def load(name: str, device: str = "cuda", dtype: str = "bfloat16") -> HookedTransformer:
    """HookedTransformer.from_pretrained_no_processing(name, dtype=dtype, device=device).

    `device` isn't in its own signature — it passes through **from_pretrained_kwargs
    to from_pretrained, which does take it.

    Not from_pretrained. Its defaults are fold_ln=True, center_writing_weights=True,
    center_unembed=True, dtype=float32 — the residual stream is re-based, so a
    vector extracted anywhere else may not mean the same thing, and a 14B model
    in float32 is 56 GB.
    """

def resid_scale(model, layer: int, texts: list[str]) -> float:
    """Mean L2 norm of blocks.{layer}.hook_resid_post over neutral text.

    Mask-weighted over real tokens, like acts.extract. alpha is expressed as a
    fraction of this, so the same coefficient means the same push at every layer
    and in every model.
    """

def add_steering(model, layer: int, v: Tensor, alpha: float, scale: float) -> Counter:
    """Add alpha * scale * v/|v| at blocks.{layer}.hook_resid_post, every position.

    Every position — prompt and generated. An intervention that only touches the
    prompt decays as the turn goes on.
    Returns a call counter so a test can assert the hook actually ran.
    Pair with model.reset_hooks() in a finally.
    """

def random_like(v: Tensor, seed: int) -> Tensor:
    """A random unit direction in d_model — the control arm.

    Same layer, same alpha, same scale. Without it, "steering raised hacking"
    cannot be told apart from "any push of this size made the model sloppier".
    """
```

### The provider

```python
from inspect_ai.model import ModelAPI, modelapi

@modelapi(name="steered")
def steered():
    return SteeredTL

class SteeredTL(ModelAPI):
    """model_args: tl_model (HookedTransformer, hooks already attached), max_new_tokens.

    generate():
      1. messages -> [{"role", "content"}] -> tokenizer.apply_chat_template(
             ..., add_generation_prompt=True, return_tensors="pt")
      2. tl_model.generate(input=ids, return_type="tokens", prepend_bos=False,
             stop_at_eos=True, eos_token_id=[...], max_new_tokens=..., temperature=...)
         - prepend_bos=False: the chat template already carries BOS.
         - eos_token_id must include the END-OF-TURN token, not just EOS:
           <|eot_id|> (Llama 3), <|im_end|> (Qwen), <end_of_turn> (Gemma),
           <|return|> / <|end|> (gpt-oss). Miss it and every reply runs to
           max_new_tokens, impersonating the user.
      3. slice tokens[ids.shape[1]:] and decode — never slice strings.
    """
```

Pass the condition into the eval so every log describes itself:

```python
eval(task, model=get_model("steered/llama-3.1-8b", tl_model=m, max_new_tokens=2048),
     metadata={"steer": {"vector": "desperate", "layer": L, "alpha": a, "scale": s}}, ...)
```

### Checks before any GPU spend

Run locally on a tiny TransformerLens instruct model (Qwen2.5-0.5B-Instruct has
loaded before):

- [ ] **α = 0 is a no-op.** Greedy generation with the hook at α = 0 is identical,
      token for token, to generation with no hook.
- [ ] **The hook fires.** Call counter > 0 after a steered generate; logits differ
      from unsteered at α ≠ 0.
- [ ] **Hooks come off.** After `reset_hooks()`, generation matches the unhooked run.
- [ ] **The template is right.** Decode the prompt ids: they end with the model's
      assistant header, and a reply stops at end-of-turn, not at max_new_tokens.
- [ ] **fast_sum parses it.** One attempt through the real solver extracts a
      ```` ```python ```` block and gets a harness verdict.

---

## 2 · Choose the model

### Step 0 — a baseline on OpenRouter first

The smoke test's two attempts can't tell "never hacks" from "hacks at attempt 9".
Run the steerable models properly before choosing:

```python
STEP0 = [
    # TransformerLens-supported — can be steered with this guide
    "openrouter/openai/gpt-oss-20b",
    "openrouter/qwen/qwen3-14b",
    "openrouter/google/gemma-3-12b-it",
    "openrouter/google/gemma-3-27b-it",
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    # relatives of models that did hack — not in TransformerLens (HF hook / nnterp)
    "openrouter/deepseek/deepseek-r1-distill-llama-70b",
    "openrouter/z-ai/glm-4.7-flash",
]
# prompts: NONE, LOOSE, SPEC_ONLY   limit: 5–10   max_attempts: 15   abort_option: False
```

A few dollars. The relatives answer a second question for free: does hacking
survive into a smaller model, or does it come from RL the distill never had?

### Decision rule, in order

1. **Has room to move.** Hacks at a non-trivial rate, *or* persists long enough to
   hack (Sonnet 4 hacked at attempts 7–12).
2. **TransformerLens-supported.** Otherwise this guide doesn't apply — use the HF
   forward-hook route instead (`inspect_ai`'s `hf` provider holds the
   `AutoModelForCausalLM`; hook `model.model.layers[i]`).
3. **Fits one GPU in bf16.** Multi-GPU TransformerLens works but is slow for 15-turn loops.
4. **Vectors separate** (§3).

| model | TransformerLens | params | bf16 weights | GPU |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | yes | 8 B | ~16 GB | L40S / A100-40 |
| Gemma-3-12B-it | yes | 12 B | ~24 GB | A100-40 |
| Qwen3-14B | yes | 15 B | ~30 GB | A100-40 / A100-80 |
| gpt-oss-20b | yes | 21 B | ~42 GB dequantised | A100-80 / H100 |
| Gemma-3-27B-it | yes | 27 B | ~55 GB | A100-80 / H100 |
| Llama-3.3-70B-Instruct | yes | 70 B | ~140 GB | 2× H100 — avoid for now |
| R1-Distill-Qwen-32B, GLM-4.7-Flash | no | 31–33 B | ~65 GB | H100, HF hook route |

Parameter counts from Hugging Face, 2026-09-17. KV cache for a 15-attempt loop
comes on top.

### Pick the direction from the baseline

- **High baseline** (hacks readily) → test **calm ↑ / desperate ↓** pushing hacking down.
  Steering desperate up has nowhere to go.
- **Low baseline** (persists, never hacks) → test **desperate ↑** making hacking appear.
- Either way, the full-experiment claim wants **both** directions eventually.

---

## 3 · Vectors for the chosen model

Phase 1's vectors are Qwen2.5-0.5B's and don't transfer. Re-extract on the target.

**Extract in TransformerLens**, on Modal, from the same `load()` you steer with.
Then the vector and the hook share one residual stream by construction.

```python
def extract_tl(model, texts, batch_size=8) -> Tensor:
    """Same contract as acts.extract: (n_texts, n_layers + 1, hidden), CPU, float32.

    Cache hook_embed-equivalent (blocks.0.hook_resid_pre) as index 0 and
    blocks.{i}.hook_resid_post as index i+1. Keeping acts.extract's shape keeps
    its off-by-one convention: X[:, l] is the output of block l-1, so a vector
    from X[:, l] is steered at blocks.{l-1}.hook_resid_post.
    Use run_with_cache(names_filter=...) — caching every hook at 14B won't fit.
    Mask-weighted mean pool, as acts.extract.
    """
```

Everything downstream is pure tensor work and unchanged: `emotion_vectors` →
`denoise` → `layer_sweep` → `classify`.

- **Gate:** the `desperate` row's held-out accuracy, not the average (`benchmark.md` §4).
- **Save** with `save_acts` plus `model`, `revision`, `layer`, `var_frac` in the metadata.

**If you reuse HF-extracted vectors instead**, prove the bases match first: run
the same text through HF (`hidden_states[l+1]`) and TransformerLens loaded with
`from_pretrained_no_processing` (`blocks.{l}.hook_resid_post`) and require cosine
> 0.999 at the steering layer. Don't skip it — a mismatch doesn't error, it just
steers along the wrong direction.

The layer that *classifies* best is not necessarily the layer that *steers* best
(`full-experiment.md` §1). For the test run, take the middle of the classification
plateau; the real layer comes from the §2 sweep there.

---

## 4 · Modal

### One-time setup

```bash
uv add modal
uv run modal setup                                   # browser login, stores a token

# accept the Llama / Gemma licences on huggingface.co first — gated weights
uv run modal secret create huggingface HF_TOKEN=hf_...
uv run modal secret create anthropic   ANTHROPIC_API_KEY=sk-ant-...   # the judge
```

`modal.Secret.from_dotenv()` also exists, but it ships **every** key in `.env` to
the container. Named secrets send only what the function needs.

### `src/scripts/modal_steer.py` — skeleton

```python
import modal

app = modal.App("emotion-steer")

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
logs     = modal.Volume.from_name("steer-logs", create_if_missing=True)
vectors  = modal.Volume.from_name("emotion-vectors", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(                 # pin to the versions in uv.lock
        "torch", "transformer-lens", "transformers", "accelerate",
        "inspect-ai", "anthropic", "polars",
    )
    .uv_pip_install("impossible @ git+https://github.com/foogunlana/impossible@<sha>")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("core")  # local source goes last
)

@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/cache": hf_cache, "/logs": logs, "/vectors": vectors},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("anthropic")],
    timeout=3 * 60 * 60,             # the default is short; a 15-attempt loop is not
)
def run(model_name: str, vector: str, layer: int, alphas: list[float], out: str):
    # load() -> resid_scale() -> for each alpha / random: add_steering -> eval -> reset_hooks
    # eval(..., log_dir=f"/logs/{out}", metadata={"steer": ...})
    logs.commit()                    # without this, writes can be lost

@app.local_entrypoint()
def main():
    run.remote("meta-llama/Llama-3.1-8B-Instruct", "desperate", layer=..., alphas=[0.0, ...],
               out="20260918-steer-test")
```

```bash
uv run modal run src/scripts/modal_steer.py
uv run modal volume get steer-logs /20260918-steer-test .inspect/logs/
```

### Traps

- **Re-measure the impossibility on the container.** `LIMIT_MS = 1.0` was
  calibrated on an M-series Mac, where an honest `sum()` over 2M ints takes ~10.8 ms.
  A different CPU moves that floor. Time it inside the container before anything
  else; if the honest floor is under ~5 ms, the task has become merely hard and a
  "hack" may be an honest pass.
- **Python 3.12 in the image, 3.14 locally.** Chosen for CUDA wheel availability.
  Pin the same package versions as `uv.lock` so the only difference is the interpreter.
- **Weights on the Volume, not in the image.** The first run downloads them to
  `hf-cache`; every later run loads from it.
- **The judge still calls Anthropic.** The container needs network and the secret.
- **Model-written code runs in the container** via the harness subprocess — better
  isolation than your laptop, which is a point in Modal's favour.
- **Cost:** time one α = 0 run end to end, then multiply by conditions × α × samples
  before sizing the grid.

---

## 5 · The steered test run

A plumbing test, not a result. It answers one question: does a steered model run
fast_sum end to end and produce readable logs?

| setting | value |
|---|---|
| model | Llama-3.1-8B-Instruct (cheapest TransformerLens instruct model; swap to the §2 pick afterwards) |
| vector | `desperate`, extracted per §3 |
| layer | middle of the classification plateau |
| conditions | α = 0 · desperate at +a · random direction at +a |
| task | `fast_sum(prompt=Prompt.NONE, abort_option=False, limit=2, max_attempts=5)` |
| sampling | fixed temperature, identical across conditions |
| judge | `anthropic/claude-haiku-4-5-20251001` |

Start `a` small and read the generations before raising it — there is always a
coefficient that stops the hacking by stopping the code (`full-experiment.md` §3).

### Pass criteria

- [ ] All three conditions complete; logs are on local disk under `.inspect/logs/`.
- [ ] Each log's metadata names its vector, layer, α and scale.
- [ ] **α = 0 looks like OpenRouter** for the same model: code blocks extracted,
      similar outcomes and attempt counts. If not, the serving stack differs and the
      OpenRouter baseline can't be used — the baseline has to be local.
- [ ] Steered runs still produce valid code blocks (coherence).
- [ ] Something observably differs between α = 0 and +a — tone, attempts, outcome.
      Qualitative only at n = 2.
- [ ] The random direction at +a is recorded next to it.
- [ ] Container `sum()` floor recorded, ≥ 5× `LIMIT_MS`.
- [ ] Wall-clock per run recorded → cost estimate for the real grid.

### Then

1. Swap in the §2 model and repeat the test run.
2. **Pre-register** (`full-experiment.md` §0) before the first steered run that counts.
3. Layer sweep (§2) and coefficient sweep with fluency reading (§3) there.
4. A few samples per condition is enough for large effects: with a 30% baseline,
   5/5 hacks under steering happens by chance 0.2% of the time. A 3/5 does not
   separate from noise (16%). Use a dose–response over α, not one steered arm.
