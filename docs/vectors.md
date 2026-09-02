---
author: claude
created: 2026-09-02
purpose: |
  Specification for the emotion-vector extraction and validation stage of the
  emotion-concepts replication. Bo writes the code himself; this holds the plan,
  the function contracts, and the traps worth knowing in advance.
context: |
  Follows docs/dataset.md. Bo is replicating Part 1 of arXiv:2604.07729 locally
  on Qwen2.5-0.5B-Instruct. Agreed order: gate the corpus, extract mean-pooled
  activations, difference-of-means on train rows only, PCA denoise against the
  neutral dialogues, logit lens, then 6-way held-out accuracy. The smoke tests
  come first because failing them is informative while passing them is not
  conclusive; the intensity and implicit tests settle that afterwards.
references:
  - https://arxiv.org/abs/2604.07729
  - docs/dataset.md
  - datasets/qwen-emotion-stories/config.yaml
  - src/core/quality.py
  - conversation context
---

> Write this as a notebook. Each fenced block below is one cell, in order.
> Bodies are left as `...` — the code is yours.

# Emotion vectors

Part 1 of [*Emotion Concepts and their Function in a Large Language Model*](https://arxiv.org/abs/2604.07729),
on a local model.

| | step | detail |
|---|---|---|
| 0 | **check the corpus is sound** | `check_corpus`, then read stories: do they name the emotion, switch language, loop, stop mid-sentence? |
| 1 | **extract activations** | all layers cached, mean-pooled over story tokens |
| 2 | **difference of means** | **train rows only**; per-emotion mean minus grand mean |
| 3 | **denoise** | PCA fit on **neutral** rows, project out top PCs to 50% variance |
| 4 | **logit lens** | sanity check on the *vectors*, no test data involved |
| 5 | **6-way accuracy** | held-out topics, chance 16.7%, plus confusion matrix |

Steps 4 and 5 are smoke tests: failing them means something is broken, passing
them does not prove the vectors are conceptual rather than lexical. The
intensity and implicit tests answer that, and come after this notebook.

**Two decisions to make deliberately rather than inherit:**

- **Which layer.** The paper works about two-thirds deep. Qwen2.5-0.5B has 24
  layers, so 16 — but cache all of them and the layer sweep is free.
- **Pooling.** Mean-pool across the story's tokens, which is what the paper does.
  This differs from the `acts()` helper in `hello-world.ipynb`, which takes the
  last token only. Last-token reads "what the model is about to say next";
  mean-pooled reads "what this passage is about". Pick one and use it on both
  sides of every comparison.

## Setup

```python
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()
)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from core.models import Model
from core.quality import report, score_story
from core.shards import read_shards
from core.utils import mentions_emotion

DATA_DIR = PROJECT_ROOT / "datasets" / "qwen-emotion-stories"
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

CONFIG = yaml.safe_load((DATA_DIR / "config.yaml").read_text(encoding="utf-8"))
rows = read_shards(DATA_DIR / "corpus")

print(f"{len(rows)} stories")
print(Counter(r["emotion"] for r in rows))
print(f"split: {Counter(r['split'] for r in rows)}")
```

## 0 · Is the corpus sound?

The gate. `names its own emotion` is the number that matters most — the prompt
forbids it, so a high rate means the model ignored the instruction and any
vector you fit could be a word direction rather than a concept.

Everything downstream assumes this step passed.

```python
# the automated checks
r = report(rows)
r["clean"], r["names_emotion"], r["flagged"]
```

```python
# ...and read some. The checks catch what they were written to catch;
# reading catches what nobody thought to look for.
#
# Look for: the emotion named outright, a mid-sentence language switch, the same
# sentence repeated, a story that stops mid-word, a story that is about the topic
# but carries no emotion at all.
...
```

## 1 · Extract activations

One vector per story: the residual stream at each layer, mean-pooled across the
story's tokens.

```python
def extract(model: Model, texts: list[str], *, batch_size: int = 16) -> torch.Tensor:
    """Mean-pooled residual stream for each text, every layer.

    Returns (n_texts, n_layers + 1, hidden). The +1 is the embedding layer,
    which `output_hidden_states=True` includes before layer 0.

    Hints:
    - `model.model(**inputs, output_hidden_states=True).hidden_states` is a tuple
      of (batch, seq, hidden) tensors, one per layer.
    - Feed the STORY TEXT ALONE, never the prompt. The prompt names the emotion;
      extraction on the text alone is what stops that leaking into the pooling.
    - Batching needs padding, and padding tokens must NOT enter the mean. Weight
      by the attention mask:
          (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
      Forgetting this silently drags every vector toward the pad embedding, by an
      amount that varies with how much padding each story got.
    - Wrap in `torch.no_grad()`.
    - Consider float32 here even though generation used fp16 — you are about to
      take means and cosines of these, where small numerical differences matter
      more than they do when sampling tokens.
    """
    ...
```

```python
model = Model(MODEL_PATH, dtype=torch.float32)
model.load_weights()

# X: (n_stories, n_layers + 1, hidden). Cache it — this is the slow step and
# every later cell is instant once it exists.
X = extract(model, [r["text"] for r in rows])
X.shape
```

```python
LAYER = 16   # ~2/3 through 24 layers. Cheap to sweep later since all are cached.

# Handy index arrays for the cells below.
emotions = sorted({r["emotion"] for r in rows if r["emotion"] != "neutral"})
is_train = torch.tensor([r["split"] == "train" for r in rows])
is_neutral = torch.tensor([r["emotion"] == "neutral" for r in rows])
labels = torch.tensor([emotions.index(r["emotion"]) if r["emotion"] in emotions else -1
                       for r in rows])

len(emotions), is_train.sum().item(), is_neutral.sum().item()
```

## 2 · Difference of means

The whole method, in two lines. The subtraction is the part that matters: without
it you get the direction that says *this is a short story*, which every emotion
shares.

```python
def emotion_vectors(X: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                    n_emotions: int, layer: int) -> torch.Tensor:
    """Per-emotion mean minus the grand mean. Returns (n_emotions, hidden).

    `mask` selects the rows to fit on — TRAIN ONLY. Fitting on everything and
    then scoring held-out rows measures how well the mean of a set describes
    that set, which is circular and always looks good.

    Hints:
    - Per-emotion mean: X[layer][(labels == k) & mask].mean(0)
    - Then subtract the mean ACROSS emotions, not across stories. Those differ:
      the across-story mean is dominated by whichever emotion has most rows.
    - Exclude neutral here — it is the denoising set, not a class.
    """
    ...
```

```python
V = emotion_vectors(X, labels, is_train, len(emotions), LAYER)
V.shape
```

## 3 · Denoise

The paper: take the principal components explaining 50% of variance across the
neutral set, and project them out of the emotion vectors.

Your neutral rows are Person/AI **dialogues**, not neutral stories — the paper's
choice. So what is being removed is assistant-transcript structure, not generic
storytelling. Worth remembering when you interpret what the denoising did.

```python
def denoise(V: torch.Tensor, neutral: torch.Tensor, var_frac: float = 0.5) -> torch.Tensor:
    """Project out the top PCs of `neutral` from each vector in V.

    Hints:
    - Centre `neutral` before the PCA, or PC1 is just its mean.
    - `torch.pca_lowrank`, or `torch.linalg.svd` on the centred matrix. Take
      enough components for the cumulative explained variance to cross var_frac.
    - Projecting out an orthonormal basis U: V - (V @ U.T) @ U
    - Sanity check afterwards: the result should be near-orthogonal to every
      component you removed.
    """
    ...


V_clean = denoise(V, X[:, LAYER][is_neutral])
# How much did it change them? A cosine near 1.0 means the denoising did little;
# near 0 means it removed most of what you had, which is worth understanding
# before trusting anything downstream.
torch.nn.functional.cosine_similarity(V, V_clean, dim=-1)
```

## 4 · Logit lens

Project each vector through the unembedding and read the top tokens. Three lines,
and it catches gross errors before you spend time puzzling over an accuracy
number.

Expect *related* tokens rather than the emotion word itself — the corpus never
names the emotion, so a vector whose top token is literally "afraid" would be
suspicious rather than reassuring.

```python
# Hint: Qwen2.5-0.5B ties its embeddings, so lm_head.weight IS the input
# embedding matrix. `model.model.get_output_embeddings().weight` is the safe way
# to ask for it either way.
#
# For each emotion vector: project, take topk, decode.
...
```

## 5 · Six-way accuracy

The smoke test. Classify held-out stories — from topics the vectors never saw —
by nearest emotion vector.

Chance is 1/6 = 16.7%. With ~100 test stories the standard error is around 4%, so
treat anything under ~25% as no signal.

```python
def classify(X_test: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Nearest emotion vector by cosine similarity. Returns predicted indices.

    Hints:
    - Normalise both sides, then one matmul: Xn @ Vn.T -> (n_test, n_emotions)
    - Subtract the SAME grand mean you used when fitting V. Skipping this
      compares a centred thing to an uncentred one, and the shared component
      dominates the cosine.
    """
    ...
```

```python
# accuracy on held-out topics
...
```

```python
# Confusion matrix — more informative than the scalar.
#
# If errors cluster between emotions that are close in valence/arousal (ashamed
# with lonely, say) rather than scattering uniformly, the vectors have structure
# even where they are wrong. That is the first hint of the geometry Part 2 is
# about, and it is visible here for free.
...
```

## What this does and does not establish

Passing means: the pipeline works, and vectors fitted on some topics generalise
to unseen ones.

It does **not** establish that the vectors are emotion concepts rather than
lexical artefacts — unless step 0 showed `names its own emotion` near zero, in
which case the shortcut was unavailable and this result is stronger.

The tests that settle it are the graded **intensity** ladders and the **implicit**
scenarios in `config.yaml`. Neither contains an emotion word, for any emotion, so
a word direction cannot score on them. That is the next notebook.
