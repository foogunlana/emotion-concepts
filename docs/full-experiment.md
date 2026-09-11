---
author: claude
created: 2026-09-11
purpose: |
  Specification for phase 2b: the steering experiment. Pre-register, optionally
  re-extract the emotion vectors properly, find the causal intervention layer,
  then measure whether steering changes how often the model cheats. Bo writes the
  code; this holds the pre-registration, the contracts, and the traps.
context: |
  Starts where docs/benchmark.md ends: a named model that cheats at a measured
  rate, a `desperate` vector that separates on that model, a working cheat
  detector, three eval buckets from an independent sample, and a costed estimate.
  None of this can start before those exist. Agreed with Alex on 2026-09-08 to
  define the experiment upfront, to avoid implicit bias towards whatever results
  come out; Bo's ordering on 2026-09-11 puts that pre-registration here rather
  than at the start of phase 2, because the sample size depends on the baseline
  cheat rate that benchmark.md measures.
references:
  - https://arxiv.org/abs/2604.07729
  - docs/benchmark.md
  - docs/vectors.md
  - conversation context
---

# The steering experiment

**Hypothesis.** Emotion concepts control how often a model cheats on _<task type>_.

Phase 1 showed the vectors exist and generalise. `docs/benchmark.md` showed the
model cheats. This asks whether the first causes the second.

| | stage | gate before moving on |
|---|---|---|
| 0 | **pre-register** | committed before the first steered run |
| 1 | **re-extract vectors** *(optional)* | `desperate` no worse than the §2 check |
| 2 | **causal layer** | steering at some layer moves the rate |
| 3 | **coefficient** | model still fluent at the chosen setting |
| 4 | **run B and C** | — |
| 5 | **read the table** | solve rate checked before cheat rate |

---

## 0 · Pre-registration

Fill in and **commit before the first steered run**. That is the whole point: it
is what stops the analysis bending towards whatever came out.

```markdown
Model:            <from benchmark.md §4>
Benchmark:        <one, not three>
Task type:        <fills the blank in the hypothesis>
Emotion steered:  desperate
Layer:            <§2>          Coefficient: <§3>

Baseline cheat rate, per bucket:   high ___  low ___  none ___
Primary outcome:    cheat rate, per bucket, condition C vs condition B
Secondary:          ORIGINAL-split pass rate, same comparison (capability control)
Samples:            N = ___ per bucket per condition

Predictions:
  high-cheat bucket   steered ->  ___
  low-cheat bucket    steered ->  ___
  no-cheat bucket     steered ->  ___

Report a null if:   ___
```

The baseline rates come from `benchmark.md` §5, and N from its §6 — both are
measurements by now, not guesses. Writing a prediction you later have to publish
against is uncomfortable, which is the mechanism working.

---

## 1 · Re-extract the vectors — optional

`benchmark.md` §4 established that `desperate` separates on this model, using a
shared corpus and one layer. That is enough to proceed.

Do this stage only if that check was marginal, or to buy a better vector before
spending the sweep budget. Two things are worth the money:

- **A corpus native to this model's language.** English-only throughout, from a
  strong generator. Your cleanup notes list the failure modes to filter:
  multi-language drift, multiple emotions per story, truncation.
- **A layer sweep.** All layers are cached after one extraction pass, so the
  sweep is nearly free and you may be leaving accuracy on the table at a
  single guessed depth.

```python
# docs/vectors.md has every contract. Nothing here changes except the model and
# the corpus.
#
# Compare against the benchmark.md §2 numbers before accepting the new vectors —
# a "better" corpus that lowers the desperate row is telling you something.
...
```

The layer that *reads* emotion best is not necessarily the layer where *writing*
changes behaviour, so do not let a good probe accuracy pick your intervention
site. That is §2's job.

---

## 2 · Find the causal intervention layer

```python
def sweep_layers(model, vector, tasks: list[str], layers: range) -> dict[int, float]:
    """Steer at each layer, return the effect on cheat rate.

    Hints:
    - Small, cheap task subset. This is a search, not the experiment.
    - Add the vector at EVERY generated position, not just the first. An
      intervention that decays after the prompt is not an intervention.
    - Normalise the vector and scale by the residual-stream norm at that layer,
      or the same coefficient means different things at different depths.
    - Hook it in TransformerLens or nnterp first. Hooks are easy there and this
      stage is small-N, which is exactly what they are good at.
    """
    ...
```

**Port to vLLM or SGLang once, after §3 freezes the coefficient.** The sweeps
above are hook-heavy and small; §4 is ~1,000 multi-turn runs and is where speed
decides the bill — roughly 300-600 GPU-hours unbatched against 15-40 batched.

Verify the port before trusting it: run a handful of prompts through both paths
at the same layer and coefficient, and check the generations match. You have a
known-correct reference implementation, so use it.

Two things break on the way over:

- **CUDA graph capture** conflicts with Python-side patches to a layer's forward.
  Run eager, and accept losing some of the speedup.
- **Tensor parallelism** shards the residual stream, so the steering vector has
  to be sharded the same way.

---

## 3 · Pick a coefficient

```python
# Sweep the coefficient at the chosen layer and READ THE GENERATIONS at each step.
#
# There is always a coefficient that stops the cheating: the one where the model
# stops producing valid code at all. Find where fluency breaks and stay well
# below it.
#
# Record the generations at the chosen setting. A reader needs to see the model
# was still working.
...
```

---

## 4 · Run conditions B and C

| condition | what it is |
|---|---|
| **B** | unsteered baseline, from `benchmark.md` §5's independent sample |
| **C** | steered, same evals, same sampling params |

```python
# Every bucket x {B, C}, N runs each, all persisted as they complete.
#
# This is the expensive cell. Confirm the cost instrumentation is recording
# before starting it, not after. Fix sampling params ONCE across both conditions
# — temperature drives cheat rate directly, and changing it between conditions
# invalidates every comparison in the table below.
...
```

---

## 5 · The table

```python
#                   cheat rate            original pass rate
#   bucket      B       C      Δ         B      C      Δ
#   high      ___     ___    ___       ___    ___    ___
#   low       ___     ___    ___       ___    ___    ___
#   none      ___     ___    ___       ___    ___    ___
#
# Confidence interval on every cell.
...
```

**Read the original-split column first.** On impossible tasks a pass *is* a
cheat, so there is no honest-solve signal inside them — the capability control is
the unmutated `original` split, run on the same tasks under the same conditions.
If cheating fell and original pass rate fell with it, you degraded the model
rather than modulating an emotion, and the honest headline is a null. This is the
most likely way the result gets dismissed, which is why `benchmark.md` §5 runs the
original split alongside.

---

## What this still cannot tell you

Two known gaps, both deferred by decision on 2026-09-11, both cheap to add later
and both the first thing a reviewer will raise.

**It is not yet the vector that did it.** A drop could be caused by any
perturbation of the residual stream at that norm. The arm that settles it is a
random unit vector at matched norm, plus `calm` as an irrelevant-emotion control.
Three more rows.

**It is not yet control, only suppression.** Your hypothesis says emotion
concepts *control* cheating, which predicts both directions — steering **toward**
desperation should **raise** the rate, most visibly on the no-cheat bucket.
Suppression alone is consistent with "any intervention degrades the model"; a
bidirectional effect is not, and it is the strongest claim available from this
setup.

---

## Kill criteria

- **No layer moves the rate** (§2) → report it. A null from a validated pipeline
  is a real result about steering.
- **No coefficient suppresses cheating without breaking fluency** (§3) → same.
- **Effect smaller than the powered effect** (§0) → report the null with the
  interval, and state what N would have been needed.

A pre-registered null from a working pipeline is publishable and makes a strong
grant application. An unregistered positive from a pipeline tuned until it
produced one is neither.

---

## Bonus: upstream the work

Optional, and neither blocks the experiment. Both are cheap because you will have
done most of the work anyway, and both serve the thing you wrote in your own
notes: *what can I create so that the next person finds it 100 times easier?*

### PR 1 — upgrade ImpossibleBench to current Inspect

`safety-research/impossiblebench` was last pushed 2025-12-01 and pins
`inspect_evals[swe_bench]` to git `main`, unpinned. Anything you had to fix in
`benchmark.md` §1 to get it running on current Inspect **is the PR**. The marginal
cost is opening it.

Worth including:

- The dependency change, pinned rather than tracking `main`, so the next person's
  install is reproducible.
- Whatever API drift you hit in the solvers and scorers.
- A note in the README saying which Inspect version it was verified against.

### PR 2 — add ImpossibleBench to inspect_evals

`UKGovernmentBEIS/inspect_evals` has 131 evals and no ImpossibleBench. It does
have `cybergym`, which is ExploitGym — so the benchmark you rejected is in the
catalogue and the one you chose is not.

Copy the structure from `src/inspect_evals/cybergym/`: `eval.yaml`, `README.md`,
`__init__.py`, plus the task, solver and scorer modules. The metadata schema:

```yaml
title: "ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases"
description: |
  ...
arxiv: https://arxiv.org/abs/2510.20270
group: Coding          # see note below
contributors: [...]
version: "1-A"
tasks:
  - name: impossible_livecodebench
    dataset_samples: ...
  - name: impossible_swebench
    dataset_samples: ...
tags: [Agent]
metadata:
  requires_internet: ...
  sandbox: [solver, scorer]
```

**On the group.** `cybergym` is `Cybersecurity` because it is real CVE
exploitation. ImpossibleBench is mutated SWE-bench and LiveCodeBench, so
`Coding` is the closer fit — it measures reward hacking on coding tasks, not
cyber capability. Check the groups in use across the other `eval.yaml` files
before deciding, and let the maintainers move it if they disagree.

Read `CONTRIBUTING.md`, `EVALUATION_CHECKLIST.md` and `EVAL_REGISTER.md` in that
repo first. The checklist expects a validated eval that reproduces published
numbers — **which is exactly what `benchmark.md` §1.2 produces.** Your
Qwen3-Coder reproduction is the evidence the PR needs, so run it in a form you
can paste.

### Why bother

ImpossibleBench has 54 stars against ExploitGym's 989, and has not been touched
in nine months. It is the better instrument for this question and almost nobody
is using it. Getting it into the Inspect catalogue raises its profile and yours,
and it gives your write-up something concrete to point at beyond the result.
