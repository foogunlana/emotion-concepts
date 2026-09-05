"""Activations, emotion vectors, and the two smoke tests.

Everything here is pure tensor work on a cached `X` — extraction is the only
step that needs the model, and it runs once. That split is deliberate: the
extraction happens wherever the GPU is, the analysis happens wherever you are.

Shapes, fixed throughout:
    X        (n_rows, n_layers + 1, hidden)   the +1 is the embedding layer
    V        (n_emotions, hidden)             one emotion vector, at one layer
    labels   (n_rows,)                        index into `emotions`, -1 for neutral
"""

from __future__ import annotations

from pathlib import Path

import torch


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

@torch.no_grad()
def extract(
    model,
    texts: list[str],
    *,
    batch_size: int = 16,
    device: str | None = None,
    progress: bool = True,
) -> torch.Tensor:
    """Mean-pooled residual stream for each text, every layer.

    Returns (n_texts, n_layers + 1, hidden) on CPU in float32.

    One forward pass per batch, all layers kept. The earlier version called this
    once for the grand mean and then once more per emotion — thirteen passes over
    the same corpus for data that never changed between them. At 0.5B that is
    merely slow; at 7B on a rented GPU it is the whole cost of the run.

    Pooling is weighted by the attention mask so padding never enters the mean.
    Without that weighting every vector drifts toward the pad embedding by an
    amount that varies with how much padding its batch happened to need — a
    per-row bias that looks like signal and correlates with story length.

    Results come back on CPU: the point is to hold all layers for the whole
    corpus at once, and that does not belong on the GPU you are renting.
    """
    device = device or model.device
    tok, net = model.tok, model.model

    # Pooling is mask-weighted, so padding side cannot affect the result. Set it
    # explicitly anyway — `load_weights` leaves it on "left" for generation, and
    # depending on that being irrelevant is a bet renewed on every edit.
    prev_padding_side = tok.padding_side
    tok.padding_side = "right"

    out = []
    try:
        for b in range(0, len(texts), batch_size):
            chunk = texts[b : b + batch_size]
            inputs = tok(chunk, padding=True, truncation=False, return_tensors="pt").to(device)

            hs = net.model(**inputs, output_hidden_states=True).hidden_states
            mask = inputs.attention_mask.unsqueeze(-1)          # (b, seq, 1)
            denom = mask.sum(1)                                  # (b, 1)

            # Pool and move to CPU inside the loop. `hidden_states` holds every
            # layer at full sequence length; at 7B with a large batch that tuple
            # alone is several GB of VRAM, and it is dead the moment it is pooled.
            pooled = torch.stack(
                [(h * mask).sum(1) / denom for h in hs], dim=1
            )                                                    # (b, L+1, hidden)
            out.append(pooled.float().cpu())

            del hs, pooled
            if progress:
                print(f"\r  extract {min(b + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
    finally:
        tok.padding_side = prev_padding_side

    if progress:
        print()
    return torch.cat(out, dim=0)


def save_acts(X: torch.Tensor, path: Path, **meta) -> Path:
    """Cache activations, half precision on disk.

    float16 halves the file for a quantity whose useful precision is nowhere
    near 11 significant digits — these become means and cosines, and the
    differences that survive that are far above fp16 resolution. It matters
    because this file has to come back over an SSH connection that dies on
    large single transfers.

    `meta` is provenance: model name, layer count, corpus size, row order. An
    activation tensor whose row order you cannot reconstruct is unusable, and
    the order lives in the corpus rather than in the tensor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"X": X.to(torch.float16), "meta": meta}, path)
    return path


def load_acts(path: Path) -> tuple[torch.Tensor, dict]:
    """Load a cache written by `save_acts`, back to float32.

    Cast up on load so everything downstream sees one dtype. Cosines of fp16
    vectors on CPU are slower than fp32, not faster.
    """
    blob = torch.load(Path(path), map_location="cpu")
    return blob["X"].float(), blob.get("meta", {})


# --------------------------------------------------------------------------
# index arrays
# --------------------------------------------------------------------------

def index_rows(rows: list[dict]) -> dict:
    """The masks and label array every later function needs, built once.

    Returns `emotions` (sorted, neutral excluded), `labels` (-1 for neutral),
    `is_train`, `is_test`, `is_neutral`.
    """
    emotions = sorted({r["emotion"] for r in rows if r["emotion"] != "neutral"})
    lookup = {e: i for i, e in enumerate(emotions)}
    return {
        "emotions": emotions,
        "labels": torch.tensor([lookup.get(r["emotion"], -1) for r in rows]),
        "is_train": torch.tensor([r["split"] == "train" for r in rows]),
        "is_test": torch.tensor([r["split"] != "train" for r in rows]),
        "is_neutral": torch.tensor([r["emotion"] == "neutral" for r in rows]),
    }


# --------------------------------------------------------------------------
# difference of means
# --------------------------------------------------------------------------

def emotion_vectors(
    X: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    n_emotions: int,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-emotion mean minus the grand mean. Returns (V, grand_mean).

    `mask` selects rows to fit on — TRAIN ONLY. Fitting on everything and then
    scoring held-out rows measures how well the mean of a set describes that
    set, which always looks good and means nothing.

    The grand mean is the mean ACROSS THE EMOTION MEANS, not across stories.
    Those are different numbers whenever the classes are unbalanced, and this
    corpus is: 91 surprised against 41 lonely. An across-story mean is dragged
    toward whichever emotion has most rows, so subtracting it leaves that
    emotion's vector short and its neighbours' long — a bias that shows up as
    one emotion never being predicted and looks exactly like a broken class.

    Neutral is excluded. It is the denoising set, not a class, and it is
    Person/AI dialogue rather than prose — leaving it in the grand mean would
    subtract transcript structure from vectors fitted on stories.

    `grand_mean` is returned because `classify` must subtract the same one. It
    is fitted, not a property of the data, and recomputing it from the test rows
    is a quiet train/test skew.
    """
    H = X[:, layer, :]                                    # (n_rows, hidden)
    means = torch.stack([
        H[(labels == k) & mask].mean(0) for k in range(n_emotions)
    ])                                                    # (n_emotions, hidden)
    grand_mean = means.mean(0)                            # across emotions
    return means - grand_mean, grand_mean


# --------------------------------------------------------------------------
# denoising
# --------------------------------------------------------------------------

def denoise(
    V: torch.Tensor,
    neutral: torch.Tensor,
    var_frac: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project out the top PCs of `neutral` from each vector in V.

    Returns (V_clean, U) where U is the (k, hidden) orthonormal basis removed,
    so you can check the result is actually orthogonal to it.

    Centre before the SVD or PC1 is just the neutral mean. Take components until
    cumulative explained variance crosses `var_frac`.
    """
    Xc = neutral - neutral.mean(0)
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)

    ratio = (S ** 2) / (S ** 2).sum()
    k = int((ratio.cumsum(0) < var_frac).sum().item()) + 1
    U = Vh[:k]                                            # (k, hidden), orthonormal

    return V - (V @ U.T) @ U, U


# --------------------------------------------------------------------------
# step 4 · logit lens
# --------------------------------------------------------------------------

@torch.no_grad()
def logit_lens(model, V: torch.Tensor, emotions: list[str], k: int = 12) -> dict[str, list[str]]:
    """Top tokens for each emotion vector, through the unembedding.

    Expect *related* tokens, not the emotion word — the corpus never names the
    emotion, so a vector whose top token is literally "afraid" would mean the
    filter leaked, not that the method worked.

    The final RMSNorm is applied first. For ranking purposes only its learned
    per-channel gain matters (the division by RMS is a scalar, and scaling a
    vector cannot reorder its dot products) — but that gain is a real reweighting
    of the channels, and skipping it reads the residual stream in the wrong basis.
    """
    net = model.model
    W_U = net.get_output_embeddings().weight              # (vocab, hidden)

    v = V.to(W_U.device, W_U.dtype)
    v = net.model.norm(v)
    logits = v @ W_U.T                                    # (n_emotions, vocab)

    top = logits.topk(k, dim=-1).indices
    return {
        e: [model.tok.decode([t]).strip() for t in top[i]]
        for i, e in enumerate(emotions)
    }


# --------------------------------------------------------------------------
# step 5 · six-way accuracy
# --------------------------------------------------------------------------

def classify(
    X_test: torch.Tensor,
    V: torch.Tensor,
    grand_mean: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Nearest emotion vector by cosine similarity. Returns predicted indices.

    Subtracts the SAME grand mean used to fit V. Without that you compare a
    centred thing to an uncentred one and the shared "this is a short story"
    component dominates every cosine, which typically pins all predictions on
    one class.
    """
    H = X_test[:, layer, :] - grand_mean
    Hn = torch.nn.functional.normalize(H, dim=-1)
    Vn = torch.nn.functional.normalize(V, dim=-1)
    return (Hn @ Vn.T).argmax(dim=-1)


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor, n: int) -> torch.Tensor:
    """Counts, rows = true, cols = predicted."""
    M = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        M[t, p] += 1
    return M


def layer_sweep(
    X: torch.Tensor,
    idx: dict,
    var_frac: float = 0.5,
    denoise_first: bool = True,
) -> list[tuple[int, float]]:
    """Held-out accuracy at every layer. Free — the activations are all cached.

    Pick the layer from this rather than inheriting "two-thirds deep" from a
    paper that used a different model. Read it as a curve: a single sharp peak
    on one layer is usually noise, a broad plateau is the real thing.
    """
    emotions, labels = idx["emotions"], idx["labels"]
    is_train, is_test, is_neutral = idx["is_train"], idx["is_test"], idx["is_neutral"]

    # Held-out emotion rows only; neutral is not a class.
    test_mask = is_test & (labels >= 0)
    y_true = labels[test_mask]

    out = []
    for layer in range(X.shape[1]):
        V, gm = emotion_vectors(X, labels, is_train, len(emotions), layer)
        if denoise_first:
            V, _ = denoise(V, X[:, layer, :][is_neutral & is_train], var_frac)
        y_pred = classify(X[test_mask], V, gm, layer)
        out.append((layer, (y_pred == y_true).float().mean().item()))
    return out
