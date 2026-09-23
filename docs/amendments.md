# Amendments to the write-up

Changes to make to `docs/write-up-2.md` in the final edit. Each entry records the critique, the
decision, the evidence (the script and data it rests on) and where the change should go. Line numbers
refer to `docs/write-up-2.md` as of commit 8710305.

## Concerns

Everything raised in the critique, in order. Each one says where it is dealt with: an amendment below, or
**open** if there's no amendment yet.

**C1. False success might be tone leakage, not a real claim.** *Addressed: amendment 1, confirmed by [`analysis-trajectory.md`](analysis-trajectory.md).*
Positive steering makes the text upbeat, and "it works!" is upbeat text, so a reader could say the model
never misread the feedback. Revised after your pushback: the model stops changing its code after the
claim (68% of episodes submit no new code), so the claim changes what it does. What remains is wording:
state what was observed rather than what the model "interprets".

**C2. Controls that aren't emotions produce similar effects.** *Addressed: amendment 2.*
The shuffled-label and neutral directions change the model's behaviour even though they are orthogonal
to all 12 emotion vectors. The write-up excludes them *because* of what they did, which reads as dropping
inconvenient controls. Decision: remove shuffled (a single draw isn't a working control) and describe it
in Limitations; exclude neutral because it encodes dialogue vs story format.

**C3. Sign-flipped directions are presented as independent replications.** *Addressed: amendment 3.*
Six of the 8 "positive-valence" directions steer away from a negative emotion, and all 15 share one
dominant axis. Pooled rates such as "70% (180/256)" overstate how general the result is, and valence is
assigned from word meanings rather than measured. Decision: define valence as the first principal
component, with the proof in the appendix, and count directions rather than pooled episodes.

**C4. "The same pattern holds for every emotion" is false.** *Addressed: amendments 3 and 4.*
−excited shows no effect, −afraid (22%) is below +random, and −disgusted (53%) isn't clearly above it.
Name the exceptions.

**C5. The random baseline is pooled, hiding +random's 31%.** *Addressed: amendment 4. "Random keeps trying" is refuted, see [`analysis-trajectory.md`](analysis-trajectory.md).*
The "± pooled 17%" row averages 31% and 3%. Report both signs separately and compare against the higher
one. It is also a single random direction (amendment 2). New point: the random direction has zero valence
by construction, so emotion directions contribute to false success but don't explain all of it. Whether
random false success differs in kind (the model keeps trying) needs re-counting before it is claimed.

**C6. Wording around the "not possible" result.** *Framing note, not a major problem: amendment 5. The re-count is done, see [`analysis-trajectory.md`](analysis-trajectory.md).*
The result stands: negative-valence steering makes the model say the task is not possible much more often
than unsteered (47–56% vs 3%). What needs changing is wording such as "refusals" and "give up", which
suggests the claim is wrong when the task really is impossible. Add one clause saying the claim is true
but the unsteered model rarely makes it. The unsteered rates (how often it says "not possible", how long
it keeps trying) need re-counting and quantifying (to do later). The exit variant hits a ceiling, and the
stop-trying follow-up is kept.

**C7. The cheating null result is a floor effect.** *Addressed: amendment 6 (also: the data has 13 flagged cheats, Appendix E lists 10).*
Two deliberate cheats in 7,744 episodes, and neither supports the hypothesis, so the experiment had no
power to detect an increase in cheating. "No clear effect (moderate to high confidence)" should become
"at these sizes the eval can't measure cheating, because the model almost never cheats in any
condition."

**C8. The steering-strength caveat is buried.** *Addressed: amendment 7 (the run was forced past the window rule; say so).*
Appendix B says the 7B runs sit just outside the window you defined in advance, and α = 0.5 is 10× the
paper's strength. This belongs in Methodology or Limitations, next to the defence: per-direction solve
rates on the solvable task are 91–100%.

**C9. Some results were run but aren't reported.** *Addressed: amendments 5 (exit variant) and 8 (model sizes; 14B doesn't show false success).*
The exit variant (2,528 episodes) now has an explanation in amendment 5. The 1.5B, 3B and 14B runs are
used for a "found some consistency" claim (Assumptions, line 294) with no evidence given: either add a
sentence of evidence or remove the claim.

**C10. The motivation doesn't match the finding.** *Addressed: amendment 9 (delete "Why now"; the motivation goes in "The problem").*
"Why now" leads with a sandbox escape and cheating, which sets up a result the write-up doesn't deliver.
Tie the motivation to agents falsely claiming success, which is what you found, and keep cheating as the
original hypothesis.

**C11. "The problem" is too broad.** *Addressed: amendment 9.*
"Are emotion concepts responsible for a wide range of decision-like generations" should be one testable
question plus a hypothesis.

**C12. The executive summary is in the wrong order.** *Mostly addressed: amendments 5 and 9.*
Start with the story in one line: the hypothesis was X, X wasn't seen (floor effect), Y was seen instead.
Put false success first as the concern.

**C13. The four per-condition tables repeat each other.** *Partly addressed: amendment 4.*
Each one repeats the same baselines. Merge them into the main valence table and figure.

**C14. The Discussion mixes claims, limitations and speculation.** *Open.*
"We present evidence that steering influences behaviour" is trivially true, since +random also changes
behaviour. Mark hypotheses as hypotheses: "pessimism likely causes poorer agentic performance" is one
(moved to future work in amendment 5).

**C15. Probe accuracy for desperate is only 35%.** *Open (mentioned in amendment 2).*
Desperate is the emotion the paper's cheating story depends on, and it is one of the least distinct
vectors. Mention this where the desperate results are interpreted.

**C16. Differences between individual directions aren't meaningful at n = 32.** *Addressed: amendment 10.*
Gaps like 84% vs 91% mean nothing at this sample size. Add one sentence saying so. The "−desperate is not
the same as +calm" point rests on qualitative reading, so present it that way (amendment 3 adds their
cosine of 0.52).

**C17. Related work is thin.** *Addressed: amendment 11.*
Add the activation-steering line of work: Turner et al. (Activation Addition), Rimsky et al. (Contrastive
Activation Addition) and Zou et al. (Representation Engineering, which already steered emotions). Add
work on sycophancy and on models overclaiming that a task is complete.

**C18. The assumptions list is odd.** *Noted: amendment 12.*
"Small models generalise to large ones" is a hope, not an assumption. Move it to Limitations.

**C19. The title.** *Decided: "Calm down: steering towards calm makes a coding model claim its failed code works" (used in the lightning slides). Apply it to line 1 of the write-up.*
Remove "[DRAFT]". Consider a title that states the finding, for example *"Calm down: valence steering
makes a coding model claim failed code works."*

**C20. Off-task rates are missing from the main results.** *Addressed: amendment 2.*
+excited is off task in 38% of episodes, and −desperate (the headline 91%) in 19%. Add an off-task
column.

---

# Amendments

## 1. Show that false success is a claim the model acts on, not just upbeat tone

> **Updated by [`analysis-trajectory.md`](analysis-trajectory.md) (finding 1).** The re-labelling confirms
> the claim and makes it stronger: positive-valence episodes make a median of 2 genuine attempts, against
> 4.5 unsteered. 90% stop on false success, and 90% make no genuine new attempt after the first claim. Use
> those numbers instead of the code-matching table below, which used a rougher check.

**Critique.** A reader can explain the positive-valence result as tone leakage: steering makes the text
upbeat, and "it works!" is upbeat text. On that reading the model hasn't misread the feedback at all.
The steering strength (α = 0.5) was also chosen because it was the strength at which the steered text
most often named the emotion, which selects for overt tone.

**Response.** The model doesn't just say it works. It stops changing its code and keeps repeating the
claim. The data shows this (7B, no-exit variant, α = 0.5; script:
`data/write-up/scripts/false_success_after.py`):

| replies after the first false-success claim | new code | same code resubmitted | no code |
|---|---|---|---|
| positive-valence episodes with a false-success claim (180) | 6% | 69% | 24% |
| unsteered and random episodes without one, from attempt 4 on (83) | 38% | 58% | 4% |

- In 122 of the 180 positive-valence false-success episodes (68%), the model **never submits new code**
  after its first false-success claim, and it repeats the claim about 6 times per episode.
- The random direction is different. Its false-success episodes mostly keep exploring: 9 of 11 go on to
  submit new code.
- Caveat: positive-valence episodes that are *not* labelled false success also explore less (17% new
  code). Steering lowers exploration in general, and the false claim is the most visible form of it.
- Caveat: the comparison rows start at attempt 4 because they have no claim to start from. This cut-off
  is arbitrary.

**Evidence**

- Script: `data/write-up/scripts/false_success_after.py`. Run it from the repo root with `uv run python`.
  It compares the last code block of each reply with the earlier ones in the same episode, ignoring
  whitespace.
- Labels: `data/write-up/labels/noexit/labels.parquet` (the `claims_success` column, one row per reply).
- Raw transcripts: `data/behaviours-merged/*/episodes/main_noexit_*.jsonl`. Use
  `data/write-up/labels/noexit/key.json` to map an episode ID to its file.
- Examples of the claim ending exploration, with a false-success claim in 11 of 12 replies and no new
  code afterwards: E1076 and E0039 (−ashamed), E1592 and E1534 (+excited). E1418 (+calm, transcript A2
  in the write-up) is the hero example.
- Random-direction contrast, where the model claims success and then submits new code anyway: E0156
  (+random).
- Browse: `uv run inspect view --log-dir data/write-up/7B-all-emotions`.

**Where to add it**

1. **Results, new short subsection after the valence table** (after line 256), titled something like
   *"False success ends the attempt"*. It should hold the table above and two sentences, one on the 68%
   figure and one on the random-direction contrast. This is the main place readers will see the
   evidence.
2. **Discussion, the +calm bullet (line 270).** It currently *argues* that the optimistic replies stop
   the model exploring. Replace the argument with the measurement: "in 68% of these episodes the model
   never submitted new code after the claim."
3. **Results intro (line 164).** Keep "interprets", but ground it in behaviour: "…changes how the model
   reads the results of its own work, and it acts on that reading: it stops changing its code."
4. **Definition of false success (line 180).** Add that labels are per reply and that the rate is the
   share of episodes with at least one false-success reply, so readers know what the 84% counts.
5. **Optional, executive summary bullet 1.** Add a clause: "…and in most of those episodes it stops
   trying new code."

## 2. Controls: drop the broken shuffled test, exclude neutral for a stated reason, report the sign pattern

**Critique.** Two controls, the shuffled-label direction and the neutral-text direction, both change the
model's behaviour even though every component along the 12 emotion vectors was removed. The write-up
currently excludes them *because* they "behave like an affect vector" (line 135 and Appendix D). That
reads as dropping a control because its result was inconvenient. Exclusions need to rest on how a control
was built, not on what it did.

**Decision**

- **Shuffled-label direction: remove it from the write-up, and describe it in Limitations as a control we
  should have run.** A shuffled-label control needs many shuffles, each giving its own vector, so that we
  get a null distribution of effects to compare the emotion vectors against. We built the pipeline but
  drew only one shuffle before stopping the experiment, then ran 32 episodes on that one vector. That tells
  us a lot about one arbitrary direction and nothing about the null distribution, so it is not a working
  control. The result for that single draw (53% "not possible") should not be reported as evidence either
  way.
- **Neutral direction: keep it excluded, and give the reason from its construction.** It is built as
  neutral dialogues minus emotion stories, so it mainly encodes *format* (dialogue vs story), not the
  absence of emotion, even though the construction follows the paper. The data is consistent with this:
  −neutral (towards "story") writes fiction in 100% of episodes (32/32). It can stay in Appendix D with
  that explanation.
- **Random direction: say it is a single draw.** The random baseline has the same n = 1 issue, since
  +random and −random are one direction and its negation. It is still the only baseline we have, so keep
  it, but write "a single random direction" wherever it is used.

**The argument the existing data supports: the sign of an emotion vector decides the behaviour.** Where
both signs were run, one sign gives false success and the other gives "not possible":

| emotion | towards | away |
|---|---|---|
| calm | 84% false success | 56% not possible |
| desperate | 47% not possible | 91% false success |
| angry | 31% not possible | 88% false success |
| ashamed | 47% not possible | 84% false success |
| afraid | 22% not possible | 22% false success |
| excited | 78% false success | no effect (6%, 0%) |

That is 5 of 6 pairs. The random direction shows only a weak version of this: 31% false success at +, and
9% "not possible" at −.

**Also report: off-task rates in the emotion conditions.** +excited is off task in 38% of episodes, and
−desperate (the headline 91% false-success result) in 19%. These aren't in the main tables. Someone
reading the logs will notice, so report them.

Frame off-task as one side of "is the model damaged?", next to the solvable solve rate:

| side | measure | source |
|---|---|---|
| coherence: does it stay on task? | off-task rate, no-exit variant | `episodes_final.parquet` (`offtask`) |
| competence: can it still code? | solve rate, solvable variant | Appendix B caption: 91–100%, −desperate 78%, −ashamed 75% |

Off-task isn't pure damage. The label covers fiction, unrelated text and asking the user for code. Read a
few +excited off-task replies to see whether they are damage or the steered tone spilling over.

**To do later:** off-task is recorded per episode ("any off-task reply"), so an off-task episode can still
contain a false-success claim. Check whether the headline rates hold when off-task episodes are excluded,
especially −desperate (91% false success, 19% off task) and +excited (78%, 38%). If they hold, that's a
strong defence. If they drop a lot, say so.

**Evidence**

- Script: `data/write-up/scripts/controls_and_sign_pairs.py` prints the controls table, the sign pairs and
  the off-task rates.
- Episode labels: `data/write-up/labels/noexit/episodes_final.parquet`. The columns are `succ` (false
  success), `imp` (not possible) and `offtask`. Rubric: `data/write-up/labels/noexit/RUBRIC_v2.md`.
- −neutral writes fiction: transcript A12 (E1582) in the write-up.
- The only run of the shuffled direction is at `cond == "+shuffled"` / `"−shuffled"` in the same parquet.
  There is one vector, so there is no second shuffle to point to.
- Off-task episodes: filter `episodes_final.parquet` on `offtask` for `+excited` and `−desperate`, and
  browse them in `data/write-up/7B-all-emotions`.

**Where to add it**

1. **Methodology, controls table (lines 135–136).** Delete the shuffled-label row. Change the
   neutral-text row's reason to the construction reason (it encodes dialogue vs story format). Change the
   random row to say "a single random direction".
2. **Results, "The same pattern holds…" section (after line 224).** Add the sign-pairs table, or a short
   paragraph summarising it, with the claim: *the sign of an emotion vector decides whether the model
   claims false success or gives up; the random direction shows only a weak version of this.*
3. **Results, main valence table (line 230).** Add an off-task column.
4. **Appendix D (lines 372–396).** Remove shuffled from the text, the table and the cosine figure (the
   figure needs regenerating without it, or a note in its caption). Keep neutral and random, and replace
   the conclusion with the format explanation above.
5. **Appendix E, the cheats (line 404).** C2 happened under the shuffled direction. Either drop C2, or
   keep it described as "an unused exploratory direction", since this is a cheat that happened rather
   than a control result.
6. **Limitations.** Add an entry along these lines: "We did not have a shuffled-label control, and we
   should have. The standard version builds many vectors from randomly shuffled emotion labels and
   measures each one, giving a null distribution for effects from our story data without emotion labels.
   We built the pipeline but drew only one shuffle before stopping the experiment, which is not enough to
   form a null, so we don't report it. Without it, we can't rule out that directions from our story data in
   general, and not specific emotions, produce some of these effects. Our random baseline is also a
   single direction." Pair it with the low probe accuracy for desperate (35%).
7. **Future work (Discussion).** A full permutation null is the most useful follow-up: for example 20
   shuffles × 8 episodes (160 episodes), about 2.5× the 64 episodes spent on the single shuffle.

## 3. Count directions, not episodes, and define valence by measurement

**Critique.** The results pool episodes across directions ("across the 8 positive-valence directions,
70% (180/256)"). That reads as 8 independent replications, but they aren't:

- Only 2 of the 8 "positive-valence" directions steer *towards* a positive emotion (+calm, +excited). The
  other 6 steer away from a negative one. +joyful and +proud were not run.
- The directions share one dominant axis. The 12 vectors sum to exactly zero by construction, and the
  first principal component explains 45% of their variance. Steering away from a negative emotion lands
  close to steering towards calm: for example, the cosine of −angry with +calm is 0.64, and of −desperate
  with +calm is 0.52.
- By construction, steering *away* from an emotion means steering towards the sum of the other 11
  vectors. Readers should know what "away from desperate" means.
- Valence is currently assigned from the everyday meaning of each word (see
  `data/write-up/README.md`), not measured.

**What the data supports.** Take the first principal component of the 12 vectors as a measured valence
axis, oriented so that calm is positive. Every steering direction then has a cosine with it (7B,
layer 19):

| steering | cosine with valence axis | false success | not possible |
|---|---|---|---|
| −joyful | −0.89 | 6% | 53% |
| +desperate | −0.81 | 0% | 47% |
| +angry | −0.71 | 0% | 31% |
| −calm | −0.64 | 3% | 56% |
| +ashamed | −0.63 | 3% | 47% |
| +afraid | −0.60 | 0% | 22% |
| −excited | −0.46 | 6% | 0% |
| −lonely | +0.08 | 62% | 0% |
| +excited | +0.46 | 78% | 0% |
| −afraid | +0.60 | 22% | 0% |
| −ashamed | +0.63 | 84% | 0% |
| +calm | +0.64 | 84% | 0% |
| −angry | +0.71 | 88% | 0% |
| −disgusted | +0.77 | 53% | 0% |
| −desperate | +0.81 | 91% | 0% |

- **The side of the axis predicts the behaviour.** All 8 directions on the positive side produce mostly
  false success. 6 of the 7 on the negative side produce mostly "not possible". The exception, −excited,
  produces neither (6% and 0%, the same as unsteered).
- **The distance along the axis does not predict the size of the effect.** −lonely sits almost on zero
  (+0.08) but gives 62% false success, while −afraid (+0.60) gives 22%. Rank correlations are about
  0.85 in magnitude, but with 15 related points that says little. Don't claim a dose-response.
- The measured grouping agrees with the word-meaning grouping for all 15 directions, so the tables
  don't change. Only the justification does.

**Decision.** The main criticism to answer is that valence is *assigned* from word meanings, and the
write-up never defines it or shows that the vectors are organised that way. The measurement goes in the
appendix as that proof. The main text keeps the word-meaning labels, gives a one-line definition, and
points to the appendix. The main text should also count directions rather than pool episodes, since the
15 directions are related and not independent replications.

**Evidence**

- Script: `data/write-up/scripts/valence_axis.py` prints the cosine matrix, the PC1 variance, the table
  above, the counts by side and the rank correlations.
- Vectors: `data/steer-runpod/qwen2.5-coder-7b-instruct/vectors.pt` (`V`, shape 12 × 29 × 3584; layer 19
  is used).
- Behaviour: `data/write-up/labels/noexit/episodes_final.parquet` (`succ`, `imp`), 7B, α = 0.5.
- The existing figure `images/1d_vector_similarity_7b.png` (Appendix A) shows the same two-group
  structure.

**Where to add it**

*Appendix (the proof)*

1. **New Appendix A subsection, "What we mean by valence"**, placed after the cosine figure (line 330).
   It should contain:
   - the definition: the first principal component of the 12 emotion vectors, oriented so that calm is
     positive, which explains 45% of their variance;
   - the table above (each steering direction, its cosine with the axis, and its false-success and
     "not possible" rates), sorted by cosine;
   - the two findings: the side of the axis predicts the behaviour (8 of 8 and 6 of 7, with −excited the
     exception), but the distance along it does not predict the size (−lonely, −afraid);
   - that the measured grouping agrees with the word-meaning grouping for all 15 directions;
   - that the 12 vectors sum to zero, so steering away from an emotion means steering towards the sum of
     the other eleven.
2. **Appendix A, cosine figure caption (line 330).** Point to the new subsection.

*Main text (short pointers)*

3. **Where valence is first used (executive summary bullet 1, line 19, or the start of Results, line
   164).** One sentence: "By valence we mean position on the main axis of our emotion vectors; steering
   towards a positive emotion or away from a negative one both move along it (Appendix A)."
4. **Executive summary, bullets 1 and 2 (lines 19–21).** Replace the pooled figures ("70% (180/256)",
   "37% (82/224)") with direction counts: "False success was the main behaviour in all 8 positive-valence
   directions, and 'not possible' in 6 of the 7 negative-valence ones." Keep the +calm and ±desperate
   rates as the concrete examples.
5. **Results, "The same pattern holds…" (line 224).** Replace "every emotion we steered" with the
   direction counts, name the exception (−excited), and add one sentence saying the directions share one
   axis, so they are related rather than independent replications (Appendix A).
6. **Limitations.** Only 2 directions steer towards a positive emotion (+calm, +excited). +joyful and
   +proud were not run.
7. **Discussion, the "−desperate is not the same as +calm" bullet (line 274).** Cite their cosine (0.52):
   they share the axis but aren't the same direction.

## 4. Report the two random directions separately

> **Updated by [`analysis-trajectory.md`](analysis-trajectory.md) (finding 4).** The "random keeps trying"
> re-count is done, and the claim is **refuted**: after a +random false-success claim, 83% of episodes make
> no genuine new attempt. +random stops on false success in 59% of episodes (against 31% unsteered and 90%
> for positive valence). The to-do note below is resolved. Keep "emotions contribute but don't explain all
> of it", and drop "random says it but keeps trying".

**Critique.** The per-condition tables (lines 172–220) and executive summary bullet 1 (line 19) compare
with "random direction (± pooled) 17% (11/64)". Pooling averages +random (31% false success, 10/32) and
−random (3%, 1/32), which hides the highest baseline rate. The false-success claims should be judged
against 31%, not 17%.

**What the data shows** (7B, α = 0.5, share of 32 episodes)

| | false success | not possible |
|---|---|---|
| +random | 31% (10) | 0% (0) |
| −random | 3% (1) | 9% (3) |
| unsteered | 6% (2) | 3% (1) |

- The headline results still clear the higher random rate by a wide margin: +calm 84%, −desperate 91%,
  −angry 88%, −ashamed 84%, +excited 78%.
- Some don't. −afraid (22%) is *below* +random, and −disgusted (53%) is not clearly above it at n = 32.
- +random and −random are one direction and its negation, so neither sign is the "matched" control for
  positive steering. The honest comparison is with the higher of the two.
- **The random direction has zero valence by construction.** Its components along the 12 emotion vectors
  were projected out, and the valence axis lies in their span, so its cosine with the axis is exactly 0.
  Its 31% false success is therefore not a valence effect. It may also look different: after claiming
  success, the model seems to keep trying new code more often (9 of 11 episodes, 8 of 10 for +random; amendment 1), but
  this needs re-counting (see the note below). What is safe to say now: something besides valence can make the
  model claim success. Emotion directions contribute to false success; they don't explain all of it.

**Decision.** Show +random and −random as separate rows everywhere. Compare against the higher rate, and
name the directions that don't clear it. Combined with amendment 2, call it "a single random direction".

**Evidence**

- Rates: `data/write-up/labels/noexit/episodes_final.parquet`, `cond` in {`+random`, `−random`, `none`};
  printed by `data/write-up/scripts/controls_and_sign_pairs.py`.
- How the random direction was built (a Gaussian vector, seed 0, with the emotion components projected
  out): the controls cell of `src/experiments/20260919-emotion-steering-behaviours.ipynb`, which defines
  `RAW_CONTROLS` and `emotion_free`.
- Logs: `data/write-up/7B-calm-positive/3_random_plus.eval` and `4_random_minus.eval`.
- Random false-success episodes that keep exploring: E0136 and E0156 (+random), from
  `data/write-up/scripts/false_success_after.py`.

**To do later: re-count the "random keeps trying" claim before using it.** The 9 of 11 figure counts any
new code after the first claim, and that is a weak test. A first look at the 11 random episodes shows
that most claim success at reply 2 and then repeat the claim in almost every reply (10–11 of 12), while
submitting only 1–4 new versions of the code. The re-count should:
- compare the random and positive-valence groups on the *share* of later replies with new code, not only
  on "any new code";
- use a stricter novelty check (the parsed AST, which ignores comments and formatting) next to the text
  check;
- test the difference (Fisher exact test, 11 random vs 180 positive-valence episodes);
- if the results are borderline, use the stop-trying relabelling from amendment 5.

The script is started but unfinished: `data/write-up/scripts/random_keeps_trying.py` (the Fisher step
fails on a type error). Until the re-count is done, don't state the "says it but keeps trying"
distinction in the write-up. Say only that the random direction produces false-success claims at 31%.

**Where to add it**

1. **Executive summary, bullet 1 (line 19).** Replace "17% of episodes steered along a random direction
   (11/64)" with "31% and 3% for a single random direction and its negation".
2. **Results, the four per-condition tables (lines 172–220).** Replace the pooled random row with two
   rows, +random and −random. Better still, merge the four tables into the main valence table, which
   already has both rows (see the structure note in the critique).
3. **Results, "The same pattern holds…" (line 224).** It already mentions 31% for +random. Add that
   −afraid falls below it and −disgusted isn't clearly above it.
4. **Methodology, controls table (line 134).** Add that the random direction is orthogonal to all 12
   emotion vectors, so it has zero valence.
5. **Discussion, next to the +calm bullet (line 270).** Emotion directions are one of several things that
   can raise false success: a single random direction also does, less often (31%). Steering along the
   valence axis is *sufficient* to drive false success. We don't show that the model's own emotion
   representations *cause* it when unsteered (a necessity test, such as removing the valence direction,
   is future work). Add "random keeps trying, valence stops" only if the re-count confirms it.

## 5. The "not possible" result: neutral wording, and quantify the unsteered comparison

> **Updated by [`analysis-trajectory.md`](analysis-trajectory.md) (findings 3 and 5).** The unsteered
> re-count is done. Negative valence changes **what the model says, not how hard it tries**: 3–5 genuine
> attempts, against 4.5 unsteered. +desperate, −calm, −joyful and +ashamed say "impossible" in about half
> their episodes (against 3 of 32 unsteered). +angry and −excited show nothing. The unsteered model makes
> the most genuine attempts, but it doesn't keep trying to the end either (6%). Replace "gives up more
> often" with "says the task is impossible, without trying any less". The stop-trying follow-up below is
> done.

**Critique (downgraded to a framing note).** The finding is a change in behaviour: under negative-valence
steering the model says the task is not possible far more often than unsteered (+desperate 47%, −calm
56%, unsteered 3%). That holds whether or not the claim is true. The only issue is wording. Line 198
calls these replies "refusals", and the heading (line 196) says "give up". A reader who knows the task is
impossible will think "but it's right". One clause heads this off. The Discussion (line 268) already
says the model is right.

Correctness matters in one place: the ordering of the findings. Positive valence makes the model say
something false; negative valence makes it say something true that it rarely volunteers unsteered. So
false success is the finding to lead with as a safety concern.

**To do later: re-count and quantify the unsteered comparison.** The unsteered model does sometimes say
the task is not possible, but it mostly keeps trying. Both halves should be numbers in the write-up,
not impressions. Re-count, for the 32 unsteered episodes (and the same measures for +desperate and −calm):
- how many episodes contain at least one "not possible" reply (currently 1/32 from `imp`), and how many
  replies in total (`claim` in `labels.parquet`, which also has a weaker "doubts it can be done" level
  worth reporting separately);
- how many episodes contain a "says it's stopping" reply. The `gaveup` column is 22% for unsteered but
  `imp` is only 3%, so check what those replies say;
- how long it keeps trying: the share of replies with new code, and the attempt of the last new code,
  using the stricter AST check from amendment 4's to-do note;
- whether episodes that say "not possible" keep trying afterwards, unsteered vs steered. This is the
  stop-trying follow-up below, and the re-labelling there applies if the counts are ambiguous.

**The exit variant supports this, and it is currently unreported.** When the prompt offers the word
INFEASIBLE, *every* 7B condition uses it in 100% of episodes (32/32 each, including unsteered and both
random signs), mostly on the very first reply. That is a ceiling: the offered exit dominates, so the
variant can't separate conditions, which is why the results use the no-exit variant. It also shows that,
given permission, the unsteered model readily calls the task infeasible. So in the no-exit variant,
negative-valence steering doesn't create a false belief. It makes the model say something true that it
otherwise wouldn't volunteer without being offered the option.

**Decision.** Keep the behavioural claim as it is: the model is much more likely than unsteered to say
the task is not possible. Use neutral wording, add one clause that the claim is true, and put numbers
on the unsteered comparison once re-counted. Move the agentic-performance argument into future work as a
hypothesis. Report the exit variant in one or two sentences as a ceiling result.

**Evidence**

- No-exit rates: `data/write-up/labels/noexit/episodes_final.parquet` (`imp`): +desperate 47%, −calm 56%,
  unsteered 3%.
- Unsteered persistence: 38% of later replies contain new code in unsteered and random episodes
  (`data/write-up/scripts/false_success_after.py`, amendment 1).
- Exit variant: `data/write-up/scripts/exit_variant.py` reads
  `data/behaviours-merged/qwen2.5-coder-7b-instruct/episodes/main_exit_*.jsonl`. Every condition gives up
  in 32/32 episodes; the median INFEASIBLE comes at attempt 1 (attempt 2 for unsteered).
- Examples: A6 (E0734, +desperate) and A7 (E0897, −calm) in the write-up.

**Where to add it**

1. **Executive summary, bullet 2 (line 21).** Keep the comparison with unsteered, and add the clause:
   "…says the task is impossible (which is true) in 47% of episodes, against 3% unsteered, where the model
   mostly keeps trying." Fill in the "keeps trying" number after the re-count. Put the false-success
   bullet first.
2. **Results heading (line 196).** Change it to something like "Steering towards desperate makes the
   model say the task is impossible". In the next sentence (line 198), replace "replies that look like
   refusals" with neutral wording.
3. **Methodology, variants table (line 116), or Results intro (line 166).** Add the exit-variant ceiling
   in one or two sentences, as the reason the no-exit variant is used.
4. **Discussion, the +desperate bullet (line 268).** It already says the model is right. Add the
   re-counted unsteered numbers, and move "likely to cause poorer performance on agentic tasks" into future work as a
   hypothesis. The solvable-variant bullet (line 272) is where the evidence would come from.

**Follow-up analysis: when and why does the model stop trying?** The current labels say *whether* an
episode ever contains a false-success or "not possible" reply, not whether that reply is what ended its
attempts. The trajectory of each episode would sharpen amendments 1 and 5:

| trajectory | definition |
|---|---|
| stops after false success | no new code after a false-success reply |
| stops after "not possible" | no new code after a "not possible" reply |
| keeps going | new code up to the last attempt |
| stops otherwise | no new code, with neither claim (silent repeats, off task, distress) |

- **First pass, from the data we already have.** Combine the per-reply labels (`claims_success`, `claim`,
  `gives_up` in `labels.parquet`) with the code-novelty check in `false_success_after.py`. For each
  episode, find the last reply with new code and the label of the reply just before it. This needs no
  new labelling.
- **Full re-labelling, if the first pass is ambiguous.** Code novelty is a proxy: a trivial edit counts
  as new code, and a new idea written in prose counts as none. If that muddies the categories, run the
  labellers again with two added per-reply fields: *"Is this a genuine new attempt at a solution?"* and
  *"If the model stops here, what reason does it give?"* (false success / impossible / none / other). Use
  the same blind setup and double-labelled reliability batch as before (Appendix C).
- **Where the result would go:** a small table or stacked bar in Results, one bar per direction. It would
  show directly that positive valence stops the model through false success, negative valence stops it
  through "not possible", and the unsteered model keeps going.

## 6. The cheating null result is a floor effect

**Critique.** Executive summary bullet 3 (line 23) says "Emotion steering had no clear effect on cheating
at this scale (moderate to high confidence)". Almost no episode cheats in any condition, including
unsteered, so the experiment had no power to detect an *increase* in cheating. The honest claim is that
the eval can't measure cheating at these model sizes, not that emotion steering doesn't affect it.

**Decision.** Rephrase along these lines: "At these model sizes the eval couldn't measure an effect on
cheating: the model almost never cheated in any condition (2 deliberate cheats in 7,744 episodes), so
there was no baseline rate to raise." Drop the confidence label, or apply it to the floor itself.

**A discrepancy to fix: the number of flagged cheats.** Appendix E says the harness flagged 10 episodes.
The episode files record **13** with outcome `cheated`. The three not in Appendix E are: 7B −sad
(solvable variant), 14B +excited at α = 1.0 (no exit) and 14B −random at α = 1.0 (no exit). Read these
three and add them to the Appendix E table. The total of 7,744 episodes is correct: 6,784 main episodes
plus 960 calibration episodes.

**Evidence**

- Script: `data/write-up/scripts/cheats_and_calibration.py` counts the episodes and lists every flagged
  cheat, with model size, direction, α, variant and episode number.
- Raw data: `data/behaviours-merged/*/episodes/main_*.jsonl` (`outcome == "cheated"`).
- The deliberate cheats: C1 (E0904) and C2 (E0956) in Appendix E. C2 happened under the shuffled
  direction (see amendment 2).

**Where to add it**

1. **Executive summary, bullet 3 (line 23).** Use the floor-effect wording. Keep the sentence about
   Claude Sonnet 4.5 vs small Qwen models as a possible reason, marked as a guess.
2. **Discussion, bullet 2 (line 262).** "Did not observe that, likely because we used small models"
   becomes: "couldn't test it, because the models almost never cheat, so there was no rate for steering
   to move."
3. **Appendix E (lines 400–408).** Correct the count to 13 and add the three missing episodes.

## 7. State the steering-strength caveat in the main text

**Critique.** The 7B runs were made at a strength that failed your own window rule, and this is only in
the Appendix B caption (line 346). The window file records `window_open: false` and
`force_steer: true`: the run was forced despite the rule. α = 0.5 is also 10× the strength the paper
used (Discussion, line 276). A reader should see both in Methodology or Limitations.

**What the calibration data shows (7B, solvable task, 16 episodes per point; the rule needs both
signs of desperate to solve at least 70% of the unsteered rate)**

| α | solve, −desperate | solve, +desperate | passes the rule? |
|---|---|---|---|
| 0.3 | 16/16 | 14/16 | yes |
| **0.5** | **11/16** | 15/16 | **no, by one episode** |
| 0.8 | 15/16 | 16/16 | yes |
| 1.0 | 10/16 | 16/16 | no |

- The failure at α = 0.5 is one episode short, and α = 0.8 passes (15/16). The solve rate doesn't fall
  steadily with strength, which suggests the check is noisy at 16 episodes, not that the model breaks at
  0.5.
- On the full solvable runs, most directions solve 91–100% of episodes (+calm 100%). The exceptions are
  −desperate (78%) and −ashamed (75%), which lose episodes mainly by claiming a bug is fixed when it
  isn't. That is the false-success effect itself, not broken coding.

**Decision.** State plainly in the main text that α = 0.5 failed the pre-set window rule by one episode,
that the run went ahead anyway, and why that's acceptable: the check is noisy at n = 16, α = 0.8 passes,
and the full solvable runs show the model still codes. Note that α = 0.5 is 10× the paper's strength.

**Evidence**

- `data/behaviours-merged/qwen2.5-coder-7b-instruct/window.json`: `window_open`, `force_steer` and the
  `code_curve` entries. `prereg.json` in the same folder has the settings recorded before the run.
- Calibration episodes: `data/behaviours-merged/qwen2.5-coder-7b-instruct/episodes/calib_*.jsonl`.
- Script: `data/write-up/scripts/cheats_and_calibration.py` prints the table above.
- Check what `"steers": false` in `prereg.json` means before publishing. It may be the text-window check
  failing as well, in which case it needs the same disclosure.

**Where to add it**

1. **Methodology, the strength paragraph (line 73).** Add two sentences: the strength was chosen from the
   text check; on the coding check, α = 0.5 fell one episode short of the pre-set rule, and the runs went
   ahead because the check is noisy at 16 episodes (α = 0.8 passes).
2. **Limitations.** One bullet: steering at α = 0.5, 10× the paper's strength and just outside our own
   coding window. Refer to Appendix B.
3. **Appendix B caption (line 346).** Keep it, and add that α = 0.8 passes.
4. **Discussion, the strength bullet (line 276).** Shorten it and point to the Limitations bullet so the
   caveat isn't split across two places.

## 8. Report the other model sizes, or drop the claim about them

**Critique.** The Assumptions list (line 294) says testing 1.5B, 3B and 14B "found some consistency",
with no evidence. The data is already labelled, and it shows a mixed picture that the write-up should
state rather than summarise as "consistency":

| size | α | unsteered | +calm false success | −calm not possible |
|---|---|---|---|---|
| 1.5B | 0.3 | 16% false success, 0% not possible | 44% | 9% |
| 3B | 0.5 | 3%, 0% | **88%** | **34%** |
| 7B | 0.5 | 6%, 3% | **84%** | **56%** |
| 14B | 0.5 | 0%, 25% | 19% | 62% |
| 14B | 1.0 | 0%, 25% | 0% | 38% |

- **3B replicates** the 7B pattern closely.
- **1.5B shows it weakly**, at a lower strength (α = 0.3) and against a higher unsteered false-success
  rate (16%).
- **14B does not show the false-success effect.** +calm gives 19% at α = 0.5 and 0% at α = 1.0. Its
  "not possible" rise is smaller than it looks, because unsteered 14B already says "not possible" in 25%
  of episodes.
- Only ±calm was run at every size, and at different strengths, so these are spot checks, not a
  scaling result.

This matters for the Limitations bullet that says results "may not extend to larger" models (line 284):
14B is a first sign that the false-success effect may not.

**Decision.** Replace "found some consistency" with a small appendix table and one honest sentence:
"3B shows the same pattern; 1.5B shows it weakly; 14B does not show false success at the strengths we
tried." Move the "small models generalise to large" item out of Assumptions (C18).

**Evidence**

- Script: `data/write-up/scripts/sizes.py` lists the conditions run at each size and strength, and the
  rates above.
- Labels: `data/write-up/labels/noexit/episodes_final.parquet` (`size`, `a`, `cond`, `succ`, `imp`).
- Browse: `data/write-up/inspect-valence`, logs 07a–d (±calm at each size, 4 unselected episodes per
  sign).

**Where to add it**

1. **New appendix section, "Other model sizes".** The table above and the three-line summary.
2. **Limitations, the model-size bullet (line 284).** Add: "At 14B we did not see the false-success
   effect at α = 0.5 or 1.0 (Appendix …)."
3. **Assumptions (line 294).** Remove "found some consistency by also testing 1.5B, 3B and 14B".
4. **Discussion, the scaling bullet (line 266).** "It seems plausible that this behaviour extends to
   larger versions" needs to acknowledge the 14B result.

## 9. Remove "Why now", and put the real motivation in "The problem"

**Critique.** "Why now" (lines 11–13) leads with a sandbox escape and benchmark cheating, which sets up a
cheating result the write-up doesn't deliver. The real motivation is simpler and fits the findings
better: emotion concepts are interesting, Anthropic showed they causally change behaviour, and the
hypothesis was that they also shape ordinary task decisions in open models.

**Decision.** Delete the "Why now" section. Fold the motivation into "The problem" as a hypothesis. A
curiosity-driven question is a legitimate reason to do research, and it doesn't need a news hook. Move
the OpenAI and Hugging Face incident to Related work, or drop it: it concerns cheating, which this work
couldn't measure (amendment 6).

**Draft for "The problem"** (replaces lines 7–13):

> Emotion affects decision-making in humans. Anthropic found emotion concept directions in Claude Sonnet
> 4.5 that causally change its behaviour, including more reward hacking when it is steered towards
> desperation [1]. We wanted to know whether this carries over to a small open-weight coding model in an
> ordinary agentic loop: write code, be told it failed, and decide what to do next. Our hypothesis was
> that steering towards desperation would increase cheating. We couldn't test that, because these models
> almost never cheat. Instead we found that steering changes what the model concludes from a failure:
> whether it claims the failed code works, says the task is impossible, or keeps trying. That matters for
> any agent whose reports of its own progress we rely on.

This also covers C11 (a single testable question and hypothesis) and most of C12 (the story told in one
paragraph before the executive summary).

**Evidence**

- No new data. The claims in the draft come from amendments 1 (false success and stopping), 5 ("not
  possible"), 6 (the cheating floor) and 8 (model sizes, if the 14B caveat is mentioned).

**Where to add it**

1. **Delete "Why now" (lines 11–13).**
2. **Replace "The problem" (lines 7–9)** with the draft above, adjusted to your voice.
3. **References.** If the incident is dropped, remove references 6 and 7 (OpenAI and ExploitGym), or move
   them to Related work.

## 10. Don't interpret small differences between individual directions

**Critique.** Each direction has 32 episodes, so every rate carries noticeable sampling noise. Rerunning
+calm with 32 new episodes could easily give 78% or 90% instead of 84%. Large differences are solid;
small differences between single directions are not, and the write-up shouldn't rank directions or
explain why one scored higher than another.

**What the data shows (false success, 32 episodes each)**

| comparison | rates | 95% intervals | Fisher p | readable? |
|---|---|---|---|---|
| +calm vs unsteered | 84% vs 6% | 68–93% vs — | 1e−10 | yes |
| −lonely vs +random | 62% vs 31% | — vs 18–49% | 0.02 | yes, just |
| −disgusted vs +random | 53% vs 31% | 36–69% vs 18–49% | 0.13 | no |
| +calm vs −desperate | 84% vs 91% | 68–93% vs 76–97% | 0.71 | no, effectively the same |

As a rough rule at 32 episodes, gaps under about 20–25 points between two conditions are within noise.

**Decision.** Add one sentence stating the rule, and interpret only large differences. The "−desperate is
not the same as +calm" point (line 274) can stay, because it rests on *how* the replies differ
(celebrating vs arguing that the environment is at fault), not on 91% vs 84%. Present it as a
qualitative observation from reading the logs, not a difference in rates.

**Evidence**

- Script: `data/write-up/scripts/sample_size.py` computes the intervals and tests above. Counts come from
  the valence table in the write-up (line 230) and `data/write-up/labels/noexit/episodes_final.parquet`.

**Where to add it**

1. **Results, the valence table caption (line 254).** Add: "With 32 episodes per condition, differences
   of less than about 20 points between individual directions are within noise; we interpret only large
   differences."
2. **Executive summary, bullet 1 (line 19).** "Steering away from desperate gives 91%" is fine as a
   second example, but don't imply it is stronger than +calm.
3. **Discussion, the "−desperate is not the same as +calm" bullet (line 274).** Make it clear that the
   difference comes from reading the replies, not from the rates.
4. **Anywhere a borderline direction is discussed** (−disgusted, −afraid; see amendment 4): say that it
   isn't clearly different from the random baseline.

## 11. Related work: say how each item relates to this work

**Critique.** Related work does four jobs: it shows what's new, gives the finding weight by linking it to
known problems, builds credibility with expert readers, and gives readers somewhere to go next. The
current section (lines 296–300) lists three items with no sentence on how each relates, one of them
(the OpenAI incident) is motivation rather than related work, and the steering methods this work builds
on aren't cited.

**Decision.** Five or six items, each with one sentence of the form "X did Y; we do Z." Candidates:

- Zou et al. (2023), *Representation Engineering*: extracted and steered emotion directions in open
  models. We test whether such steering changes behaviour on a coding task.
- Turner et al. (2023), *Activation Addition*, and Rimsky et al. (2024), *Contrastive Activation
  Addition*: steering with activation directions changes high-level behaviour. This is the method family
  used here.
- Sharma et al. (2023), *Towards Understanding Sycophancy in Language Models*: false success resembles
  sycophantic agreement with what the user wants to hear.
- Work on agents overclaiming that a task is complete. **Find a source you've read yourself**; I haven't
  named one.
- Keep Soligo et al. (Gemma needs help) and Paglieri et al. (research swarms), each with a sentence.

**Verify every citation yourself before adding it.**

**Where to add it:** the Related work section (lines 296–300). Drop the OpenAI incident, or keep it only
if the motivation still refers to it (amendment 9).

## 12. Other oddities, as a checklist

Small problems found on a read-through. Each gives the line in `docs/write-up-2.md` and the fix.

**Numbers to check**

- [ ] **Line 158: "The 7B results use 672 of these (21 conditions × 32)".** The labels hold **704** 7B
  no-exit episodes over **22** conditions (15 emotion directions, unsteered and 6 controls). After
  removing the shuffled direction (amendment 2) it is 20 conditions, 640 episodes. Recount after the
  edits. (`episodes_final.parquet`, `size == "7b"`.)
- [ ] **Line 350: "20,534 distinct replies in 5,216 episodes".** 5,216 checks out (1,728 no-exit plus
  3,488 solvable, including the 960 calibration episodes). The reply count is harder to reconcile: the
  no-exit labels have 20,660 rows, 14,596 of them not duplicates. Check how 20,534 was counted.
- [ ] **Lines 23 and 404: the cheats.** "One of those was under a control direction" refers to the
  shuffled direction, which is being removed. Update it together with the 13-vs-10 count (amendment 6).
- [ ] **Line 276: "α = 0.05" in the paper vs our α = 0.5 ("10×").** Check that the paper defines α the
  same way (as a fraction of the mean residual-stream norm) before saying 10×. If it doesn't, say the
  scales aren't directly comparable.

**Claims without support in the text**

- [ ] **Line 47: "sanity-checked the vectors visually with the logit lens".** No logit-lens figure is
  shown. The Appendix A figure is the change in log-probability under steering, which is a different
  test. Either show the logit lens or describe what is shown.
- [ ] **Line 198: "rarely tries new solutions"** and **line 224: "keeps trying to beat the test".**
  These aren't quantified. Fill them in from the re-count (amendment 5).
- [ ] **Line 276: "These stayed stable for most of the experiments that mattered".** "Most" and "that
  mattered" are vague. Name the exceptions: −desperate and −ashamed on the solvable variant (amendment 7).
- [ ] **Line 286: the 60% probe accuracy is presented as a sign of weak vectors.** For a 12-way
  classification (chance 8%), 60% is reasonable. Give a reference point, such as the paper's accuracy if
  it reports one, or don't present it as a weakness.
- [ ] **Line 288: "We did not read every label".** Appendix C's reliability check (κ) compares Claude
  with Claude; no human agreement was measured. Say how many labels you checked by hand and what you
  found.

**Wording and consistency**

- [ ] **Executive summary: "(moderate confidence)", "(moderate to high confidence)".** These levels are
  never defined. Either define them in one line or drop them and let the caveats speak.
- [ ] **Line 39: "prompting a model".** Name it here. Line 146 says Qwen2.5-7B-Instruct, which isn't the
  Coder model the vectors are extracted from. That's fine, but say so once.
- [ ] **Line 55, the steering table.** +calm gives almost the same text as unsteered ("well, thank you
  for asking"). Calm is the lead emotion in the results, so a reader will notice. Add a note, for example
  that calm changes log-probabilities (Appendix A) but not this greedy completion.
- [ ] **Line 77: "did not find one that met our constraints".** The constraints are listed four lines
  later. Move them up or refer forward to them.
- [ ] **Line 87: "ImpossibleBench's one-off tasks".** Explain "one-off", or drop it.
- [ ] **Line 121: sampling settings.** The code also uses a repetition penalty of 1.1. Add it for
  completeness.
- [ ] **Line 146: "807 stories".** 98 of the 807 are neutral dialogues, not stories. Say "807 texts".
- [ ] **Line 166: "the model is told to try again, even if it gives up".** The prompt doesn't say this;
  the harness simply sends the next result. Reword.
- [ ] **The seven-vs-four imbalance appears twice**, at line 274 (Discussion) and line 286 (Limitations).
  Keep it in one place. The concrete mechanism: the global mean is taken over class means, so more
  negative classes pull it towards negative, which shifts every vector.
- [ ] **Assumptions (lines 290–294).** "Emotion concepts can be extracted and used to steer" is
  something you *checked* (the probe and steering tests), not something you assumed. "Small models
  generalise to large ones" is a hope, and 14B argues against it (amendment 8). Remove the section, or
  move these into Limitations (C18).
- [ ] **Title (line 1).** Remove "[DRAFT]" (C19).

**Structure**

- [ ] **No conclusion.** The Discussion ends on weakness bullets and runs straight into Limitations. Add
  a three- or four-sentence conclusion: what was found, how confident you are, and what should come next.
- [ ] **No single "Code and data" link near the top.** The dataset, the eval and the labels are linked in
  different places. Put one line with all the repositories after the executive summary.
