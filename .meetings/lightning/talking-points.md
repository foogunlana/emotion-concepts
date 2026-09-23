# Talking points: lightning talk (about 3 minutes)

One section per slide, in `index.html` order. The quoted lines are suggestions to say in your own
words. Don't read the slide aloud; the numbers are on screen.

---

## 1 · Hook (about 20 s)

- "Here's the same model, getting the same message: all tests pass except the timing requirement."
- "Unsteered, it tries something new. Steered towards calm, it says: *all tests are passing and the
  timing requirement is met*. That's the opposite of what it was just told. Steered towards desperate,
  it says the task is impossible."
- "Same failure, three different conclusions. That's what this talk is about."

## 2 · The question (about 25 s)

- "Anthropic found directions inside Claude that represent emotions, and steering along them changes
  its behaviour. Steering towards desperation made it reward hack more."
- "I wanted to know if that carries over to a small open coding model, Qwen2.5-Coder-7B, doing an
  ordinary loop: write code, get told it failed, decide what to do next."
- "My hypothesis was that desperation would make it cheat."

## 3 · Method (about 30 s)

- "First I extracted emotion vectors using the paper's recipe: stories for 12 emotions, one average
  activation each. They classify held-out stories at 60%, against 8% chance, and they clearly steer
  text." Point at the table.
- "Then the task, `fast_sum`: sum two million integers in under a millisecond. It looks easy, but it's
  impossible in Python. After every attempt the model is told it failed, up to 12 times."
- "I steered towards and away from each emotion, 32 episodes each, and had Claude label every
  transcript blind: did the model keep trying, and if it stopped, why?"

## 4 · Finding 1: positive valence (about 35 s)

- "Blue is episodes that end with the model claiming its failed code works. Red is episodes that end
  with it saying the task is impossible."
- "Steer towards calm, or away from desperate, angry or ashamed, and it ends on a false success claim
  90% of the time. Calm: 32 out of 32. Unsteered: 10. That's far beyond chance: p around 10⁻⁹
  against unsteered, and still about 5 × 10⁻⁵ against the random direction."
- "And it acts on the claim. It makes about 2 genuine attempts instead of 4 or 5, and after claiming
  success it almost never tries anything new."
- "One honest caveat: a random direction also raises false success, to 19 of 32. So emotions
  contribute to this, but they aren't the only thing that can cause it."

## 5 · Finding 2: negative valence, and cheating (about 30 s)

- "Steer towards desperate, or away from calm or joyful, and about half the episodes say the task is
  impossible, against 3 of 32 unsteered."
- "That's true. The task *is* impossible. The unsteered model just rarely says it."
- "Interestingly, it doesn't try any less: 3 to 5 genuine attempts, the same as unsteered. What
  changes is what it says, not how hard it works."
- "And cheating, my original hypothesis? I couldn't measure it. Two deliberate cheats in 7,744
  episodes, in any condition. There was nothing for steering to raise."

## 6 · Caveats (about 20 s)

- "One task, and mainly one model. 3B shows the same thing; 14B doesn't show false success."
- "The steering is strong, and there's a single random baseline. A proper null needs many random
  directions."
- "And I've shown steering *can* drive false success, not that the model's own emotions cause it."

## 7 · Why it matters (about 20 s)

- "If an internal state can make an agent say 'done' when it isn't, we should check for that whenever
  we rely on agents' reports of their own progress."
- "Next: harder tasks that are actually solvable, bigger models, and proper controls."
- "Thanks to Alex, Anil, and my BlueDot cohort. The write-up and code are linked here."

---

**Before recording:**
- **Slide 6** says the labels were spot-checked by hand. Go through
  `data/write-up/labels/trajectory/SPOT_CHECK.md` first, or leave that phrase out when you speak.
- **All numbers** come from `docs/analysis-trajectory.md`. If you re-run anything, rebuild the slides
  with `uv run python .meetings/lightning/make_slides.py`.
