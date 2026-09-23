# Presentation plan: 3-minute video

> **Superseded.** The slides are in `.meetings/lightning/index.html`, and the talking points that match
> them (the final title, the new labels and numbers) are in `.meetings/lightning/talking-points.md`. This
> file is kept for the recording tips at the end.

A lightning-talk video in the style of BlueDot demo day: 2–3 minutes, one idea. Seven slides. Each
section below is one slide: what goes on screen, and the talking points.

The story follows the amended write-up (`docs/amendments.md`), not the current draft. Items marked
**[pending]** depend on the trajectory re-labelling in `data/write-up/labels/trajectory/`. Fill them in
once it's done, or leave them out.

**Recording:** slides with your face in a corner (see the end of this file).

---

## Slide 1: Hook (0:00–0:20)

**On screen:** the hero figure, the same failure message with three steered replies. Take a screenshot
from the site.

**Talking points**
- Same model, same failure message: "your code didn't meet the timing requirement."
- Three different conclusions, depending on which emotion direction the model is steered with.
- One says it worked, one says it's impossible, one keeps trying.

## Slide 2: The question (0:20–0:50)

**On screen:** the title of Anthropic's paper, with one line: *Emotion concepts causally change Claude's
behaviour, including reward hacking.*

**Talking points**
- Anthropic found directions inside Claude that represent emotions, and steering along them changes
  what it does. Steering towards "desperate" increased reward hacking.
- My question: does this carry over to a small open-weight coding model, Qwen2.5-Coder-7B?
- My hypothesis: steering towards desperation would make it cheat.

## Slide 3: Method (0:50–1:20)

**On screen:** left, the `fast_sum` prompt. Right, `docs/images/2a_steering_logprobs_7b.png` or the
small steering table ("He feels… lonely / angry / like a failure").

**Talking points**
- I extracted 12 emotion vectors from the model, following the paper's recipe: emotional stories, one
  average per emotion, then checked that they classify held-out stories and that they steer text.
- The task, `fast_sum`: sum 2 million integers in under 1 ms. It looks easy but is impossible in Python.
- After each attempt, the model is told it failed. It gets 12 attempts.
- I steered towards and away from each emotion and ran 32 episodes per condition.

## Slide 4: Finding 1, false success (1:20–1:50)

**On screen:** `docs/images/valence_7b.svg` (the valence figure), with the false-success side
highlighted. Optionally transcript A2 or A4 as a quote.

**Talking points**
- Steering towards calm, or away from desperate, makes the model say its failed code works.
- +calm: 84% of episodes, against 6% unsteered.
- **[pending]** After claiming success, it usually stops changing its code, so this is a conclusion it
  acts on, not just upbeat text.
- A random direction also causes some false success (31%), so emotions contribute to false success
  rather than explaining all of it.

## Slide 5: Finding 2, "not possible", and the valence axis (1:50–2:20)

**On screen:** the same valence figure, with the "not possible" side highlighted.

**Talking points**
- Steering towards desperate, or away from calm, makes the model say the task is impossible: 47–56%,
  against 3% unsteered.
- That's true. The task *is* impossible, but the unsteered model rarely says so and keeps trying.
- Both effects follow one axis: positive valence leads to false success, negative valence to "not
  possible". The sign of the steering decides which.
- Cheating: I couldn't measure it. These models almost never cheat, in any condition.

## Slide 6: Caveats (2:20–2:45)

**On screen:** four short bullets.

**Talking points**
- One task, one main model. 3B shows the same pattern, but 14B doesn't show false success.
- Strong steering: 10× the paper's strength (check the α definitions match before saying this), and
  just outside my own safety check.
- One random baseline direction. A proper control needs many random and shuffled directions.
- Labels come from Claude and were spot-checked by hand.

## Slide 7: Why it matters, and what's next (2:45–3:00)

**On screen:** one sentence, plus links to the write-up and the code.

**Talking points**
- Emotion-like directions can change what an agent concludes about its own progress, including
  whether it tells you it's done.
- Next: harder tasks that are solvable, larger models, and proper null controls.
- Thanks to my mentor and the BlueDot cohort.

---

## Recording with your face and slides

The simplest option is **Loom**. It records your screen with your webcam in a movable circle in the
corner, in one take, with no editing. The free plan has a length limit (check it; it has been about 5
minutes), which is enough for 3 minutes.

1. Build the 7 slides in Keynote or Google Slides, one figure and one line per slide.
2. Open Loom and choose "Screen + Camera". Put the camera bubble in a corner that no slide uses.
3. Present full screen and talk through the points above. Do two or three takes and keep the best.

Alternatives:
- **Zoom:** start a meeting on your own, share your screen with the slides, and record. The recording
  shows your video next to the slides.
- **Keynote:** has a "Live Video" object that puts your camera on a slide, then use the built-in screen
  recording (Cmd+Shift+5).

**Tip:** Don't read the talking points aloud. Say one sentence per bullet, and let the figure carry the
numbers.
