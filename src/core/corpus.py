"""Building prompts and generating the story corpus.

Lives here rather than in the notebook so it can be driven from a terminal for
long runs. The notebook imports from this module — one implementation.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Sequence

import torch

from core.shards import write_jsonl
from core.types import Prompt, Story
from core.models import Model
from core.prompts import SYSTEM, render
from core.utils import mentions_emotion, trim_to_sentence


def build_instruction(emotion: str, topic: str) -> str:
    """The user turn for one (emotion, topic) pair — see core.prompts.

    The emotion word appears in the INSTRUCTION and, if the model complies,
    nowhere in the story. Activations are extracted from the story alone.
    """
    return render(emotion, topic)


def build_prompts(
    emotions: Sequence[str],
    topics: Sequence[str],
    n_per_pair: int = 4,
    include_neutral: bool = True,
) -> list[Prompt]:
    """Cross product of emotions x topics x n_per_pair, plus neutral.

    `index` is local to the (emotion, topic) pair, never a running counter, so
    adding an emotion or a topic only ever appends work — shards already
    generated stay valid.
    """
    labels = [*emotions, "neutral"] if include_neutral else list(emotions)
    return [
        Prompt(e, t, n, instruction=build_instruction(e, t))
        for e, t, n in itertools.product(labels, topics, range(1, n_per_pair + 1))
    ]


def chunked(seq: Sequence, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def batch_seed(run_seed: int, batch: Sequence[Prompt]) -> int:
    """Seed from the batch's identity, not its position.

    Keyed on emotion AND topic AND index: seeding on the batch index alone gave
    every shard the same RNG stream, and emotions whose prompts differ by one
    word produced near-identical stories. Keying on emotion alone fixes that
    across shards but leaves every batch WITHIN a shard starting from the same
    stream, so position j of each batch draws identical randomness.

    blake2b, not hash(): the builtin is salted per process.
    """
    p = batch[0]
    key = f"{run_seed}:{p.emotion}:{p.topic}:{p.index}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**31)


def generate_stories(
    model: Model,
    prompts: Sequence[Prompt],
    *,
    batch_size: int = 16,
    seed: int = 0,
    max_new_tokens: int = 512,
    system: str | None = SYSTEM,
    verbose: bool = True,
) -> Iterator[Story]:
    prompts = list(prompts)
    for i, batch in enumerate(chunked(prompts, batch_size)):
        seed_for_batch = batch_seed(seed, batch)
        torch.manual_seed(seed_for_batch)

        t0 = time.perf_counter()
        texts = model.gen(batch, max_new_tokens=max_new_tokens, sample=True, system=system)
        if verbose:
            done = i * batch_size + len(batch)
            print(f"{done}/{len(prompts)}  {time.perf_counter() - t0:.1f}s", flush=True)

        for p, text in zip(batch, texts):
            # Trim BEFORE mentions_emotion: a half-sentence the cap cut off can
            # name the forbidden word and fail a story that is otherwise clean.
            text = trim_to_sentence(text.strip())
            yield Story(
                emotion=p.emotion,
                topic=p.topic,
                index=p.index,
                prompt=p.instruction,
                text=text,
                # Per BATCH, not per story: one generate() call samples the whole
                # batch from a single RNG stream, so 16 rows share this value.
                # Story identity is (emotion, topic, index), not the seed.
                batch_seed=seed_for_batch,
                mentions_emotion=mentions_emotion(text, p.emotion),
            )


def generate_corpus(
    model: Model,
    out_dir: Path,
    *,
    emotions: Sequence[str],
    topics: Sequence[str],
    n_per_pair: int = 4,
    limit: int | None = None,
    batch_size: int = 16,
    seed: int = 0,
    max_new_tokens: int = 512,
    system: str | None = SYSTEM,
    include_neutral: bool = True,
) -> dict[str, int]:
    """One shard per emotion, skipping any that already exist.

    The shard is the unit of resume, so a crash costs one emotion.

    `limit` writes to <emotion>.sample.jsonl, which read_shards filters out —
    otherwise a 2-story eyeball run leaves complete-looking shards that every
    later run skips, and you silently end up with a 26-story corpus.
    """
    labels = [*emotions, "neutral"] if include_neutral else list(emotions)
    counts: dict[str, int] = {}

    for emotion in labels:
        name = f"{emotion}.sample" if limit else emotion
        path = out_dir / "corpus" / f"{name}.jsonl"
        # Resume protects real shards only. A sample IS the thing you rerun
        # after changing a prompt, so skipping it just makes you delete files
        # by hand and wonder why nothing changed.
        if path.exists() and not limit:
            print(f"skip {name}", flush=True)
            continue

        prompts = build_prompts([emotion], topics, n_per_pair, include_neutral=False)
        if limit:
            prompts = prompts[:limit]

        print(f"{name}: {len(prompts)} prompts", flush=True)
        stories = generate_stories(model, prompts, batch_size=batch_size, seed=seed,
                                   max_new_tokens=max_new_tokens, system=system)
        counts[name] = write_jsonl((asdict(s) for s in stories), path)
        print(f"wrote {name}: {counts[name]}", flush=True)

    return counts
