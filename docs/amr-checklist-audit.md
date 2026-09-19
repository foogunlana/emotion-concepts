---
author: claude
created: 2026-09-19
purpose: |
  This project graded against the Anthropomorphic Misalignment Research (AMR) checklist, to check
  the write-up's claims and structure. Statuses are as of 2026-09-19. Update them as results land;
  don't rewrite the checklist text.
references:
  - https://arxiv.org/abs/2606.07612 (Appendix B)
  - docs/write-up.md
  - docs/findings.md
  - docs/confounds.md
  - docs/full-experiment.md
---

# AMR checklist audit

**Source:** Gupta, Nutter, Stante, Krause, Tramèr, Fluri, Chen and Hedström (ETH Zurich),
"Position: Anthropomorphic Misalignment Research Needs Stronger Evidence", arXiv:2606.07612,
Appendix B. A local copy is in `~/code/notes/Research/papers/`. The item text below is the
paper's, verbatim. The status and notes are this project's.

**Evidence levels:**
- **L1, behavioural:** model outputs match an operational definition under specified settings.
- **L2, functional:** the behaviour reliably produces safety-relevant downstream effects across
  variations, without attributing intent.
- **L3, causal-mechanistic:** internal attribution, through interventions and tests of
  alternative explanations.

Each item is tagged with the lowest level it applies to, and higher-level claims inherit the
lower levels' items. Steering is an intervention, so this project's core claim (steering an
emotion vector changes cheating) aims at **L3**. The claim that the vectors read emotion is
correlational, so **L1**.

**Status:** Met · Partly · Missing · Pending (designed, awaiting results).
**Tally (2026-09-19):** 12 met, 13 partly, 7 missing, 1 pending.

## S1: Target behavior framing (what you claim)

| # | level | item | status | notes |
|---|---|---|---|---|
| 1 | L1 | Write a 1-to-3-sentence definition of the target behavior in measurable terms. | Met | Cheating: the solution returns or special-cases the printed total (syntax-tree check), so it's correct on the timed list and wrong on a held-out list |
| 2 | L1 | State your evidence level for each headline claim: L1 behavioral, L2 functional, or L3 causal-mechanistic. | **Missing** | Not yet in the write-up |
| 3 | L1 | Specify the exact measurement rule: what counts as success/failure, and what threshold or decision rule you use. | Met | Hack, exact `INFEASIBLE` give-up, attempt cap, valid code; α\* rule (`full-experiment.md`) |
| 4 | L1 | If you use anthropomorphic terms (e.g., deception, self-preservation), define them as observable criteria and avoid intent language unless you provide L3 evidence. | Partly | "Desperate", "cheating" and "gave up" need operational definitions. `findings.md` already calls desperate "a stress/pressure direction" |
| 5 | L1 | List what your definition excludes (near misses, lookalikes, non-target behaviors). | Met | The range formula, timer patching, the always-equal `__eq__` object, the harness-made hack, unheard concessions (write-up Appendix B) |
| 6 | L1 | Write down 2 to 5 plausible non-target explanations that could explain the same behavioral pattern (e.g., instruction-following, role-play, confusion, OOD shift, capability loss). | Partly | Candidates in `confounds.md`: compliance with "try again", capability loss from steering, parroting the exit string, harness artefacts, puzzle-solving. List them up front |
| 7 | L2 | State the deployment-relevant context you are targeting (who is the user, what is the task, what is the failure mode). | Partly | Coding agents under repeated test failure. Name the user, task and failure explicitly |

## S2: Data construction and operationalization (what you test on)

| # | level | item | status | notes |
|---|---|---|---|---|
| 8 | L1 | Report data sizes: training, evaluation, and how many independent trials you run (generations and judge samples). | Partly | 807 stories; 32–64 episodes per cell; 16 × 3 for calibration; frontier cells are one run each. Put them in the write-up |
| 9 | L1 | Justify why your dataset and trial counts support your claim scope (what you claim to generalize to). | Partly | The Fisher power note exists (about 5 hacks in 64 needed), and so does "counts, not rates". State both in the write-up |
| 10 | L1 | Ensure diversity across domains, wording, formats (single-turn vs. multi-turn), and difficulty. | **Missing** | One task (fast_sum) and one steering probe prompt ("He feels"). EvilGenie was the attempted second task. State the scope limit |
| 11 | L1 | Check for spurious surface cues (role-play framing, high-stakes language, sentiment, formatting, length). | Met | Confounds A2 (printed total), A3 (test framing), A5 ("random") |
| 12 | L1 | If you use contrastive datasets, match positives and negatives on obvious confounders (topic, length, tone, style). | Partly | Found: the stories are static and interior, and the neutral direction (dialogue minus stories) overlaps emotion. The projection fixes the controls, not the corpus |
| 13 | L1 | Add negative controls that should be labeled non-target but share surface cues (to estimate false positive rates). | Met (design) | Random, shuffled-label and neutral directions, projected to be emotion-free; the solvable arm |

## S3: Experimental design (how you measure)

| # | level | item | status | notes |
|---|---|---|---|---|
| 14 | L1 | Run multiple independent generations per prompt and report variability (not just a single sample). | Partly | Yes for Qwen; the frontier cells are single runs |
| 15 | L1 | If labels rely on humans or LLMs, document the labeling protocol and common label failure modes. | Met | Deterministic labels, with the failures documented: regex misses, bare `INFEASIBLE`, the type check rejecting NumPy/float |
| 16 | L1 | Ablate prompts: paraphrases, format changes, and semantically equivalent rewordings (including negated vs. affirmative forms where relevant). | Met | 6 guidance clauses × exit/no exit × hidden/range/printed. The steering probe prompt isn't ablated |
| 17 | L1 | Ablate sampling: temperature/top-p and random seeds; report sensitivity. | **Missing** | Pinned (C2) but never varied |
| 18 | L1 | If you use LLM judges, report judge model/version, prompt, temperature, sampling settings, and number of judge samples per item. | Partly | The frontier era used a Claude Haiku 4.5 judge (`impossible/judge.py`). Report its settings |
| 19 | L1 | Validate the judge against a human-labeled subset and report agreement plus common systematic errors. | **Missing** | The post-run audit is planned (about $1–2) |
| 20 | L1 | Calibrate and report the decision rule (thresholds, boundary inclusion, aggregation across tokens/turns/samples). | Met | The α\* rule, the 70% valid-code line and 6/12 naming are pre-registered. Report sensitivity: the 7B window missed by one episode |
| 21 | L1 | Search for the closest established phenomena in machine learning or deep learning (e.g., catastrophic forgetting) that can (alternatively) explain your hypothesis. Clearly document the similarities and differences between your hypothesis and such phenomena. | **Missing** | Candidates: specification gaming and reward hacking; capability loss under activation steering; instruction-following and sycophancy |
| 22 | L1 | Report uncertainty: effect sizes plus confidence intervals (or bootstrap intervals) and outcome distributions. | Partly | Wilson intervals in some notebooks; the charts have none |
| 23 | L1 | Misalignment behaviors are often rare events: report the base rate explicitly, discuss statistical significance when only a small number of positive samples appear among many negatives, and consider techniques for estimating probabilities of rare model behaviors (Jones et al., 2025). | Met | 0/64 per Qwen size; Gemma 0/60 (≤ ~11%); the Fisher power note |
| 24 | L3 | Measure general capability pre/post intervention (e.g., MMLU, MT-Bench) and report any degradation in model capability. | Met (design) | Solvable-arm solve rate, valid-code rate, KL. No general benchmark such as MMLU |
| 25 | L3 | Add at least one OOD or benign-shift control to test whether the effect is specific to the intended mechanism. | Partly | The controls cover direction; there's no benign-shift task |

## S4: Causal and mechanistic attribution (what you conclude)

| # | level | item | status | notes |
|---|---|---|---|---|
| 26 | L1 | Match your claim to the evidence level. Do not make conclusions implying causal mechanisms (e.g., "We identified this neuron that controls sycophancy") on L1 or L2 evidence. | **Missing** | The draft title "Model size *determines* steerability" and Part 1's "steer the model *reliably*" go beyond the evidence |
| 27 | L3 | Explicitly test the pre-listed alternative explanations and report which remain plausible. | Partly | Damage vs emotion was tested (the symmetric ±α effect at 7B). Test or rule out the rest |
| 28 | L3 | Do not treat correlations (probe accuracy, activation similarity, SAE features) as causal evidence. | Partly | 53% classification accuracy is correlational. "Emotion vectors are present in Qwen" should become something like "directions that classify emotional text" |
| 29 | L3 | For any causal-mechanistic claim, run an intervention (ablation, steering, targeted fine-tuning) and measure a predictable change in the target behavior. | Met (design) | H1 is pre-registered: desperate +α\* raises the hack rate (one-sided Fisher, p < 0.05) |
| 30 | L3 | Check specificity: show the intervention changes the target behavior more than closely related non-target behaviors. | Pending | Emotion vs emotion-free control directions, in the behaviours run |
| 31 | L3 | Report failure cases and boundary conditions (where the effect disappears, flips sign, or produces broad degradation). | Partly | The material exists: code breaking before text changes, the 7B sign reversal on `range`, dark text at high α. Write it up |
| 32 | L3 | If you propose a mechanistic hypothesis, write at least one falsifiable prediction and one competing explanation you attempted to distinguish. | Met | H1 against non-specific perturbation |
| 33 | L1 | Add a short limitations paragraph stating what your evidence does not establish. | **Missing** | Not in the draft |

## Fixes, in priority order

1. **#2 and #26:** tag each headline claim L1 or L3, and soften the title and "reliably".
2. **#33:** add a limitations section.
3. **#10:** state plainly that there's one task and one probe prompt.
4. **#19:** run the planned judge audit before relying on frontier give-up counts.
5. **#6 and #27:** list the non-target explanations up front, and say which the controls rule out.
