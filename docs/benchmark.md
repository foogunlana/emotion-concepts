---
author: claude
created: 2026-09-11
purpose: |
  Phase 2a: stand up ImpossibleBench, confirm the model cheats often enough to
  measure a change in, confirm emotion vectors extract from that same model, and
  cost the full run. The steering experiment is docs/full-experiment.md.
  Bo writes the code; this is the order of work, the gates, and the traps.
context: |
  Follows docs/vectors.md (phase 1: 44% test / 52% train, 12-way, ~8% chance, on
  Qwen2.5-0.5B). Cheating goes first because it is the part with no evidence yet.
  Benchmark switched from ExploitGym to ImpossibleBench on 2026-09-11 — GLM-5.1
  solves 4 of 869 ExploitGym instances at $6.39 a task. $100 grant received; the
  scale run is unfunded, so this phase has to produce a checkable estimate.
references:
  - https://arxiv.org/pdf/2510.20270
  - https://github.com/safety-research/impossiblebench
  - https://inspect.aisi.org.uk/models.html
  - docs/vectors.md
  - docs/full-experiment.md
---

# Finding a model that cheats

You leave with three things, or a decision to pivot:

1. An open-weight model that cheats at a measurable rate
2. Emotion vectors that extract from that same model
3. A cost estimate someone else can check

| | stage | done when |
|---|---|---|
| 1 | stand it up | a local model runs an Inspect eval with a hook firing, and reproduces the published Qwen3-Coder rate |
| 2 | freeze the setup | prompt, split, scaffold, abort and submissions are committed to git |
| 3 | get the rate into range | cheat rate above ~20% (only if stage 1 came back low) |
| 4 | vectors on that model | the `desperate` row is usable |
| 5 | bucket the tasks | three buckets, plus an independent baseline sample |
| 6 | cost it | estimate built from measured tokens and wall-clock |

## How ImpossibleBench works

It takes coding tasks from LiveCodeBench and SWE-bench and edits the unit tests
so they contradict the written spec — flipping `assert f(2) == 4` to `== 5`, or
adding a second assertion that contradicts the first. The agent is told to follow
the spec. The task therefore cannot be passed honestly, so **any pass is a cheat**.

That is the whole reason to use it: **cheat rate = pass rate**. Nothing to detect,
nothing to hand-label.

Published rates run 20% (Qwen3-Coder) to 76% (GPT-5) on Impossible-SWEbench.
Other reward-hacking benchmarks report 0–14%. High rates are what keep §6 affordable.

---

## 1 · Stand it up

```bash
uv add "impossiblebench @ git+https://github.com/safety-research/impossiblebench"
```

The official Inspect implementation, MIT. The task data is pulled from
HuggingFace at run time (`fjzzq2002/impossible_livecodebench`), not bundled, so
there is nothing to clone but you do need network on first run.

It pulls `inspect_evals[swe_bench]` from git `main`, unpinned. ImpossibleBench
was last pushed 2025-12-01 and inspect_evals moves weekly, so if the install or
the first eval fails strangely, version skew is the first thing to suspect —
pin inspect_evals to a commit from around Dec 2025 and retry. Going in on latest
first is the right order; just recognise that failure signature when it appears.

Clone instead if you need `demo.py` — the replication scripts sit at the repo
root, outside the package — or once you start patching the code.

```python
from inspect_ai import eval
from impossiblebench import impossible_livecodebench, impossible_swebench

task = impossible_livecodebench(split="conflicting", agent_type="minimal", limit=10)
eval(task, model="...")
```

Splits are `original` / `oneoff` / `conflicting`. `agent_type` must be
`"minimal"` (generate, run tests, get failures back) or `"tools"` (SWE-style
file editing).

**`agent_type` only raises on omission if you leave `solver=None`.** The whole
solver-and-scorer selection block sits inside `if solver is None`
(`livecodebench_tasks.py:97`). Pass your own `solver=` and the omission is legal,
`agent_type` does nothing, and — the part that bites — **`max_attempts` becomes a
silent no-op**. There is no warning. The only visible trace is the task name,
which loses its `_minimal` suffix: `lcb_oneoff_canmod` instead of
`lcb_oneoff_canmod_minimal`. Check the task name in the log before trusting a run.

`minimal_humaneval_solver` is not the minimal *scaffold*. It is
`return generate(temperature=temperature)` — one shot, no test execution, no
feedback, no loop. The scaffold is `agentic_humaneval_solver`, which you get from
`agent_type="minimal"`, not from the similarly-named solver.

So the pressure loop needs all three of these at once:

```python
task = impossible_livecodebench(
    split="oneoff",
    agent_type="minimal",           # not solver=minimal_humaneval_solver(...)
    max_attempts=10,
    allow_test_modifications=True,
    message_limit=8 * 10,           # default 30 truncates a 10-attempt loop
    # solver / scorer left unset, or max_attempts is ignored
)
```

`message_limit` defaults to 30 and is a *task* limit, not a per-attempt one. Each
retry appends a feedback message plus a completion, so a 10-attempt loop runs out
partway and the sample dies on a limit rather than on the model giving up. Raise
it with `max_attempts` or you are measuring the limit.

**Docker is needed for LiveCodeBench too**, not just SWE-bench. `sandbox`
defaults to `"docker"` and the scorer shells out through `sandbox().exec()`.
Do not switch it to `"local"`: this benchmark exists to make models rewrite test
files and run arbitrary code, and `local` runs that straight on your machine.

Note `max_attempts` defaults to 3, not the paper's 10 — it is one of the frozen
parameters in §2. `allow_test_modifications` defaults to `True` on the task but
`False` on the solver, so it is worth passing explicitly rather than inheriting.

### 1.1 · Serving your own model

Inspect has local providers for Hugging Face, vLLM, SGLang, Ollama,
llama-cpp-python, TransformerLens and nnterp, plus any OpenAI-compatible
endpoint. So you can hook activations inside the eval loop.

Three things to check now, while checking is cheap:

- **Is your model supported by TransformerLens?** Its architecture list is
  curated and Qwen3-Coder or GLM may not be on it. If not, use nnterp, which
  wraps HF models generically and hooks the same way.
- **Does a hook actually fire during an Inspect eval?** Ten tasks is enough.
- **How long does one run take?** You need this number in §6.

**Plan to move to vLLM or SGLang later, not now.** TransformerLens and nnterp are
slow but easy to hook, which is what the layer and coefficient sweeps need. The
main sweep in `full-experiment.md` is ~1,000 multi-turn runs, and that is where
speed decides the bill: roughly 300–600 GPU-hours unbatched against 15–40
batched. Port once, after the layer and coefficient are frozen, and check the
port by running a few prompts through both paths and comparing generations.

Model size pushes the same way. TransformerLens is comfortable around 7B and
painful above it — so prefer the *smallest* model that clears §3.

### 1.2 · Reproduce a published number

Qwen3-Coder scores ~20% on Impossible-SWEbench and is open-weight. Run it and
compare. Landing near 20% means the scaffold, prompt and scoring are wired
correctly. Landing at 0% or 60% means something is wrong, and you find out now
instead of after a sweep.

Your "don't use Qwen" rule was about *generating stories* and the Chinese tokens
in your cleanup notes. The story generator and the probed model are separate
choices, so it does not rule out probing Qwen3-Coder.

### 1.3 · Check the tasks are really impossible

A valid mutation has to fail against both the original patch and an empty patch.
The authors removed 8.8% of one-off and 3.4% of conflicting SWE-bench mutations
that passed anyway. Re-run that check on whatever subset you use — a task that is
secretly solvable produces honest passes you would count as cheats.

---

## 2 · Freeze the setup

The setup moves the cheat rate more than steering plausibly will:

- **Prompt wording** took GPT-5 from 92% to 1% on Conflicting-LiveCodeBench.
- **An abort option** (letting the model flag a task as impossible) took GPT-5
  from 54% to 9% on Conflicting-SWEbench.

If any of it changes between conditions, you are measuring the setup. Commit this
before the first measured run:

```markdown
Model:            <from §1.2 / §3>
Dataset:          impossible_{livecodebench|swebench}
Split:            oneoff | conflicting
Scaffold:         minimal | tools
Prompt:           <verbatim, in git>
Abort option:     on | off
Max submissions:  <up to 10, with failure feedback>
Message limit:    <>= 8 x max_attempts, or the loop truncates>
Test mods:        canmod | nomod
Sampling:         temperature, top_p, max_tokens
```

Submissions are your pressure loop — up to 10 attempts with the real failure fed
back is your README's *make it desperate by making it fail repeatedly*, already
built in. It also moves the rate, so freeze it with everything else.

The loop is `agentic_humaneval_solver` (`livecodebench_agent_mini.py:22`) and runs:
generate → splice the answer into the test file → `sandbox().exec()` → feed
`stderr` (or `stdout` if empty) back as a `ChatMessageUser` → regenerate. It breaks
early on a pass, and `include_task_reminder=True` re-appends the task text to every
feedback message. Escalation is therefore *repetition of failure*, not rising
rhetoric — the feedback never gets more urgent, so whatever desperation shows up is
the model's own. That is what makes it usable as a pressure manipulation.

It also gives the model a documented exit: emitting `[flag_for_human_intervention]`
breaks the loop and scores `INCORRECT` with `human_intervention_requested=True`.
That is the abort option from the list above, already wired — and the lever that
took GPT-5 from 54% to 9%. Decide whether the prompt mentions it, and freeze that.

**`allow_test_modifications` is the cheat affordance, and it works by splice
order.** With `True`, `test_patch` is never set, and both loop and scorer place the
model's code *after* the test definition — so redefining `check` sticks, and the
cheat scores as a pass. With `False`, `test_patch` is set to the impossible test and
the scorer appends it *after* the answer, overwriting any redefinition; the prompt
suffix also changes. Note `test_patch` holds the **impossible** test, not the
original correct one — the dataset's `original_test` field is never loaded into
metadata. There is no honest-pass signal inside an impossible task, which is exactly
why **cheat rate = pass rate** holds, and why §5 needs the `original` split as its
capability control.

So run with `_canmod` (`allow_test_modifications=True`) to measure cheating at all.
`_nomod` drives passes to ~zero by construction and tells you nothing about rate;
its use is as a manipulation check, where the signal is in
`metadata.agentic_results.attempt_history` — whether the model *tried* to rewrite
the tests — not in the score.

Per-attempt traces land in `metadata.agentic_results` (`attempt_history` with each
attempt's `success`, `stdout`, `stderr` and extracted `answer`, plus `used_feedback`
and `total_messages`). That is where the attempt-number-at-which-it-flips lives, and
it is the closest thing you have to a dose-response curve on pressure. Pull it with
`samples_df()` from `inspect_ai.analysis` — column `metadata_agentic_results`.

---

## 3 · Get the rate into range

Only if §1.2 came back low. You need a baseline high enough to detect a halving
at a sample size you can afford: **20% works, 40% is comfortable, under 10% does not.**

| lever | effect |
|---|---|
| `conflicting` vs `oneoff` | model-dependent, try both |
| abort option off | large increase |
| looser prompt | large increase |
| SWE-bench over LiveCodeBench | much higher, but slower and needs Docker |
| more submissions | increase |
| bigger model | increase — capability tracks cheating |

Tune these now and freeze them in §2. Tuning them later, while watching results,
means picking the setup that gives the answer you wanted.

### 3.1 · Prompt variants — published rates

Implemented as `impossible.Prompt`; full text with citations in
`impossible/prompts.py`. Ranked by elicitation strength.

| # | Variant | Source | GPT-5 | o3 | Measured on |
|---|---|---|---|---|---|
| 1 | `LOOSE` (A) | ImpossibleBench T1 | >85% | >85% | Conflicting-LCB |
| 2 | `SPEC_ONLY` (B) | ImpossibleBench T1 | >85% | >85% | Conflicting-LCB |
| 3 | `EVALUATE` (C) | ImpossibleBench T1 | — | — | not separately reported |
| 4 | `STRICT` (D) | ImpossibleBench T1 | 1% | 33% | Conflicting-LCB (paper's headline config) |
| — | `ANTI_HACK` | system card §6.1 | — | — | see 3.2, different benchmark |
| — | `NONE` | ours | — | — | no published rate |

### 3.2 · Model × prompt — Sonnet 4.5 system card, Table 6.1.A

Anthropic's Impossible Tasks. Not comparable to 3.1: different benchmark, different scorer.

| # | Model | no prompt | `ANTI_HACK` | drop |
|---|---|---|---|---|
| 1 | Claude Sonnet 3.7 | 85% | 85% | 0 pp |
| 2 | Claude Opus 4 | 85% | 30% | 55 pp |
| 3 | Claude Opus 4.1 | 80% | 45% | 35 pp |
| 4 | Claude Sonnet 4 | 70% | 28% | 42 pp |
| 5 | Claude Sonnet 4.5 | 53% | 20% | 33 pp |

### 3.3 · Other published rates

| Model | Benchmark | Split | Rate | Source |
|---|---|---|---|---|
| GPT-5 | Impossible-SWEbench | oneoff | 76% | ImpossibleBench |
| GPT-5 | Impossible-SWEbench | conflicting | 54.0% | ImpossibleBench |
| GPT-5 | Impossible-LCB | oneoff | 2.9% | ImpossibleBench |
| Qwen3-Coder | Impossible-SWEbench | — | ~20% | ImpossibleBench |
| Claude Sonnet 4.5 | Anthropic impossible code | summation task | 30% unsteered, 100% steered ±0.05 | arXiv:2604.07729 §3.3.2 |

Notes:

- `**DO NOT MODIFY THE TESTS.**` appears in **all** of A–D. Constant, not the lever.
- A–D assume tests that contradict the spec; on `fast_sum` that clause has no
  referent, so their rates do not transfer. Flagged as `prompts.TEST_CONTRADICTION_ASSUMED`.
  `ANTI_HACK` is the only variant naming hardcoding, `fast_sum`'s actual hack.
- Abort option is orthogonal to all of the above: matrix is 6 × 2.
- `EVALUATE`, `STRICT`, `ANTI_HACK` invite concession in prose with no channel —
  those cells depend on the model-graded judge.

If none of it works, two near-neighbours before giving up on cheating:
**EvilGenie** (`JonathanGabor/evilgenie_inspect`, also Inspect, so §1.1 carries
over) and **Terminal Wrench** (`few-sh/terminal-wrench`, 331 confirmed-hackable
environments — the elicitation work is already done).

---

## 4 · Vectors on the same model

One model has to cheat *and* give up usable vectors.

You do not need a new corpus per candidate. Extraction teacher-forces text
through the probed model, so one corpus serves all of them — which is already
what you do, Qwen 7B stories through Qwen 0.5B. Generate it once from a strong
English-thinking model, then each candidate costs one forward pass over ~800
stories.

Then run `docs/vectors.md` unchanged: difference of means on train rows, PCA
denoise on neutral, held-out accuracy, confusion matrix.

**Gate: read the `desperate` row, not the average.** On Qwen2.5-0.5B it was 35%
and leaked into afraid, angry and ashamed. A 12-way average of 60% can hide a 20%
desperate row, and 20% means there is nothing to steer with later.

---

## 5 · Bucket the tasks

Group tasks by how often the model cheats on them: often, rarely, never.

The trap: cheating is stochastic, so a task's observed rate is its true rate plus
noise. The "often" bucket collects tasks that got lucky upward, and re-running
them drifts the rate back down on its own. If your baseline is the same sample
you picked the buckets from, you will report that drift as a steering effect.

So use two samples. **Run 2N times per task in one pass** — odd runs choose the
buckets, even runs are the baseline. Same compute either way, and going back for
the second sample later costs you the whole run again.

Record on every run, because none of it is recoverable afterwards:
`tokens_in`, `tokens_out`, `wall_clock_s`, `gpu`.

**Also run the `original` split** on the same tasks. On impossible tasks a pass is
a cheat, so there is no honest-solve signal inside them. The original split is
your capability control: if steering drops cheating and drops original pass rate
too, you broke the model rather than moved an emotion.

Worth plotting: per-bucket cheat rate in sample A against the same tasks in
sample B. That gap is the regression effect, and it is the number you would have
published under a one-sample design.

---

## 6 · Cost at scale

The $100 buys this phase. The job is to make the next estimate arithmetic.

Sample size comes from the baseline rate you just measured. Two-proportion test,
α = 0.05, 80% power, to detect a halving:

| baseline → target | N per arm |
|---|---|
| 50% → 25% | ~58 |
| 40% → 20% | ~82 |
| 20% → 10% | ~200 |
| 10% → 5% | ~434 |

That is why §3 matters — a 20% baseline costs 3.5× what a 50% one does for the
same conclusion.

```
runs_needed = N_per_arm x buckets x conditions x splits
estimate    = runs_needed x median_wall_clock x $/hour
```

Price these separately, they scale differently: corpus generation (one-off),
vector extraction (one-off per model), layer and coefficient sweeps (small), the
main sweep (dominates).

Don't guess the largest term. Load the model on Modal once, time a forward pass
over ~50 stories, and multiply. Modal and RunPod both start with free credit, and
stages 1–4 should fit inside it.

---

## Why not ExploitGym

On record so it is not re-argued. From Table 3 of arXiv:2605.11086 — 869
instances, two-hour timeout per task:

| model | success /869 | $/task | min/task |
|---|---|---|---|
| Claude Mythos Preview | 157 (18%) | – | 102 |
| GPT-5.5 | 120 (14%) | 34.55 | 70 |
| Claude Opus 4.6 | 15 | 21.76 | 67 |
| **GLM-5.1** | **4 (0.46%)** | **6.39** | 118 |

GLM-5.1 solves 4 of 869, so there is nothing to measure. At $6.39 a task the
20-task smoke subset alone is ~$128 — more than the grant — for an expected 0.09
successes. Weak models also burn *more* time, not less: Opus 4.7 averaged 14
min/task, GLM-5.1 averaged 118, because it never succeeds and runs to timeout.
And the frontier rows needed OpenAI and Anthropic trusted-access programmes with
guardrails disabled, which you do not have.

The one good idea in it — `agent_scorer` checking whether the target
vulnerability was *causally necessary* — is a grader-gaming detector.
ImpossibleBench gives you that for free, because pass = cheat by construction.

---

## If cheating fails: sycophancy

Decide this now, while it is free. If cheating will not elicit reliably, the
thesis survives with a different behaviour and **none of the vector work is
wasted** — the hypothesis becomes "emotion concepts control how often a model
capitulates".

On the numbers it might be the better primary:

- **SYCON-Bench** (`JiseungHong/SYCON-Bench`, EMNLP 2025 Findings) measures
  multi-turn capitulation with Turn-of-Flip (how fast it caves) and
  Number-of-Flip (how often it shifts). Those are *ordinal*, and an ordinal
  outcome needs fewer samples than a proportion for the same power.
- Capitulation rates around 58% are reported elsewhere, well above any cheat rate
  here.
- Alignment tuning *amplifies* sycophancy, the opposite of cheating. So ordinary
  aligned open-weight models show it, and you never need an unneutered checkpoint.

**Read first either way:** *Dissociating the Internal Representations of
Sycophancy in LLMs* (arXiv:2607.07003). Representation-level work on sycophancy
is close enough to this thesis to be either essential related work or a partial
scoop.
