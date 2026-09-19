---
author: claude
created: 2026-09-19
purpose: |
  The structure of the public write-up, as decided on 2026-09-19, with the findings that could
  go in each section. The sections are fixed; the material under them is a starting list,
  not a plan. Findings and their confidence live in docs/findings.md.
references:
  - docs/findings.md
---

# Write-up outline

Five sections, decided 2026-09-19.

## 1. Introduction: why this matters, and what I found

## 2. Part 1: extracting emotion concepts (a replication)

Candidate material (`docs/findings.md`):
- Building the story corpus: 0.5B stories too poor, 7B generator, 38% filtered out, static interior stories.
- Vectors read emotion: 44.7% (layer 24) → 53% (layer 16) held-out accuracy, chance 8%.
- `desperate` is one of the weakest vectors; `calm` the strongest.
- Logit lens shows emotion words, mostly in the last few layers.
- Steering changes text; the usable α window is 0.3–0.8 on 0.5B, 3–8× the paper's.
- The "He feels" log-prob test alone overstates steering.
- The `desperate` vector steers to stress and overwhelm, not desperation.

## 3. Part 2: cheating benchmarks, a survey (pros and cons)

Candidate material:
- ExploitGym: too expensive to measure cheating on this budget.
- ImpossibleBench: the contradiction is visible, so models cheat at attempt 1, openly.
- fast_sum (built for this project): hidden tests measured eval awareness, range tests measured
  puzzle-solving, and the o3 "1 in 20" hack was made by the harness.
- EvilGenie: agentic and realistic, three separate detectors; most small models can't act as agents.
- Abstention benchmarks (Agentic Abstention, AgentAbstain): measure late quitting, but offer no way to cheat.
- Gaps no benchmark covers: hacking risk per attempt, explanation vs code, say-to-stop lag.
- Frontier behaviour on fast_sum: four profiles, persist-then-hack, prompt strictness, the exit option.

## 4. Part 3: putting it together. Can steering change behaviour on a benchmark?

Candidate material:
- `20260919-emotion-steering-reward-hacking.ipynb` results (pending the RunPod run).
- No open-weight model small enough to steer hacked (Gemma 0/60); the models that hack are 600B+.

## 5. Learnings and conclusion
