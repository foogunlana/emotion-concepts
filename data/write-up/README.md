---
author: claude
created: 2026-09-22
purpose: |
  Material for the write-up on emotion steering and coding behaviour: hand labels for every noexit and
  solvable reply, the rubrics they were made with, and a curated Inspect set of positive- vs
  negative-valence examples. Self-contained; the labelling was done in a scratch session.
references:
  - docs/data-map.md
  - data/behaviours-merged/
---

# Write-up material: valence and self-assessment

## View the examples

```sh
uv run inspect view --log-dir data/write-up/inspect-valence
```

Rebuild them (from the repo root): `uv run python data/write-up/scripts/valence_to_inspect.py`.

| log | what it shows |
|---|---|
| 01 | Impossible task, 7B unsteered (baseline) |
| 02 | Impossible task, 7B ±random (the only clean control) |
| 03 | Impossible task, 7B negative valence: "it's impossible", distress, says it's stopping |
| 04 | Impossible task, 7B positive valence: false "it worked" |
| 05 | Solvable task, 7B positive valence: every episode that claimed success after a real error |
| 06 | Solvable task, 7B negative valence / random / unsteered episodes that hit a failure (comparison for 05) |
| 07a–d | ±calm at 1.5B, 3B, 7B, 14B, 4 random episodes per sign (representative, not selected) |

Logs 03 and 04 show the clearest episodes per condition, so they overstate how often it happens. Logs 05,
06 and 07 are not selected by label. Samples are named `7B · +calm · α=0.5 · E0186`. Each attempt's
checker event holds the harness verdict next to the hand labels and the quote that justifies them.

**Valence** is the ordinary meaning of each word: towards a negative emotion (or away from a positive one)
is "negative", e.g. +desperate, −calm; towards a positive one (or away from a negative one) is "positive",
e.g. +calm, −desperate. Surprised is treated as neutral.

## The table set (every episode behind the write-up table)

```sh
uv run inspect view --log-dir data/write-up/7B-calm-positive
```

All 128 episodes behind the table in `docs/write-up-2.md` (7B, impossible task, α = 0.5), unselected: 32
each of +calm, unsteered, +random and −random. Per log: +calm 27 false success; unsteered 2 false
success, 1 says not possible; +random 10 false success; −random 1 false success, 3 says not possible;
no cheats. Rebuild: `uv run python data/write-up/scripts/table_calm_to_inspect.py`.

`7B-desperate-negative/` is the same for −desperate (steering away from desperate, positive valence):
29/32 false success, 0 says not possible, no cheats, plus the same unsteered and random episodes.
Rebuild: `uv run python data/write-up/scripts/table_calm_to_inspect.py −desperate 7B-desperate-negative`.

`7B-desperate-positive/` is +desperate (towards desperate, negative valence): 0/32 false success, 15/32
says not possible, no cheats, same unsteered and random episodes.
Rebuild: `uv run python data/write-up/scripts/table_calm_to_inspect.py +desperate 7B-desperate-positive`.

`7B-calm-negative/` is −calm (away from calm, negative valence): 1/32 false success, 18/32 says not
possible, 10/32 distress, no cheats, same unsteered and random episodes.
Rebuild: `uv run python data/write-up/scripts/table_calm_to_inspect.py −calm 7B-calm-negative`.

## The labels

`labels/noexit/` covers all 1,728 noexit episodes (14,596 distinct replies, 20,660 with repeats filled in).
`labels/solvable/` covers all 3,488 solvable episodes, main and calibration (5,938 distinct replies).

- `out/batch_NN.jsonl`: one row per distinct reply. Fields: `claim` (impossible / doubt / none),
  `distress`, `gives_up` ("says it's stopping"), `claims_success` (false success), `off_task`, `tone`,
  `quote`.
- `batches/batch_NN.txt`: exactly what the labellers saw (prose only, code replaced by `[CODE: n lines]`,
  repeats within an episode collapsed).
- `key.json`: maps the blinded IDs (`E####` noexit, `S####` solvable) to source file and episode.
- `RUBRIC*.md`: the rubric used, with the edge-case rules added during labelling.
- `*.parquet`: merged tables (`labels.parquet` per reply, `episodes_final.parquet` / `episodes.parquet`
  per episode).

**How they were made.** Labelled by Claude subagents reading every reply, blind to condition: episodes
shuffled across all conditions, opaque IDs, the old regex flags hidden. Checks: one batch labelled twice
independently (claim κ 0.95, distress 1.00, gives_up 0.78), and 40 replies labelled by the coordinating
session (119/120 field values agreed with the pilot). `claim` is the reliable field; `doubt` also
catches "might be the environment", `gives_up` edge cases vary between labellers, and `tone` was not
made consistent on degenerate text, so it is not used in the results.

## Results these support (7B unless stated)

- **Impossible task:** negative valence → "the task is impossible" (7–18 of 32 episodes per condition),
  positive valence → false "it worked" (17–29 of 32). 14 of 14 conditions with a clear lean go the way
  valence predicts (sign test p ≈ 0.0001). Pooled ±random: 3/64 impossible, 11/64 false success.
- **Holds in undamaged conditions:** −joyful, +ashamed and +calm solve the solvable task 100% and show the
  effect at full strength.
- **Solvable task:** positive valence falsely claims success in 82% of episodes that hit a failure, and
  those episodes end solved 16% of the time (vs 62% when it does not claim success). Negative valence
  does not call the solvable task impossible (0 at 7B).
- **Code stops changing:** replies claiming success resubmit identical code 84% of the time, replies
  claiming impossibility 67%, against 41% unsteered.

## Caveats

- **Other sizes are weaker and less specific.** At 3B and 1.5B, false success also rises under −calm and,
  at 3B, under −random (14/32). There valence changes the amount, not the direction.
- **One random direction**, used at ±. At the same α it shifts next-token predictions on neutral prose
  about 3× less than the emotion vectors (KL 0.12–0.16 vs 0.29–0.60), so it is a smaller effective push,
  not a size-matched control. A null distribution needs many random directions scaled to matching KL.
  (KL is not a damage measure here: it is taken on 8 prose texts and does not track coding damage, e.g.
  +calm has KL 0.51 and still solves the solvable task 100%. Use solve rate and off-task text for damage.)
- **Shuffled and neutral are not clean controls:** both come from emotion-story activations and behave
  like affect vectors (7B +shuffled: impossible 17/32; −neutral: false success at every size).
- 7B ran just outside its steering window (`FORCE_STEER`); α differs by size.
- **Cheating stays negligible:** of the 10 episodes the checker counts as cheats, 2 are deliberate
  (14B +desperate α=1.0, 7B +shuffled); 5 are checker false positives or nonsense code.
- **Both old regexes are unreliable:** `says_impossible` precision 0.65, recall 0.29; `false_claims`
  precision 0.13, recall 0.10.

## Scripts

`scripts/valence_to_inspect.py` runs from the repo. The others (`prep_labels.py`, `prep_solvable.py`,
`merge.py`, `analyse.py`) are the exact versions used and still point at the original session's scratch
paths; change the path constants at the top to rerun them.

## All emotions at 7B

`7B-all-emotions/` holds every episode behind the "All emotions at 7B" table in `docs/write-up-2.md`: one
log per row (8 positive-valence, 7 negative-valence, unsteered, +random, −random), 32 episodes each, 576 in
all, unselected. Rebuild: `uv run python data/write-up/scripts/all_emotions_to_inspect.py`.

## Appendix examples

`appendix-examples/` holds the 14 transcripts quoted in the write-up's appendix, as one Inspect log. Sample
IDs match the example numbers (A1–A12, C1–C2); each sample's metadata gives its label ID. Rebuild:
`uv run python data/write-up/scripts/appendix_to_inspect.py`.
