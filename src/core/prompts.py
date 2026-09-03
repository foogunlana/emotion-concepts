"""Generation prompts, taken from the paper's appendix.

Sources: arXiv:2604.07729, appendix "Dataset generation" — the emotional
stories system prompt and the neutral dialogues system prompt.

The load-bearing instruction is the prohibition on naming the emotion. Without
it the stories say "she was furious", a difference-of-means picks up the token
rather than the concept, and every downstream test can be passed by a word
direction. With it, `mentions_emotion` becomes a compliance check on the corpus
instead of a caveat on the results.

Three additions to the paper's prompt, kept only because each one measurably
fixed something. Everything else that was tried is listed after them, so the
same ground does not get retrodden.

A system prompt licenses the fiction frame (SYSTEM, below); without it
Qwen2.5-0.5B declined 6% of the corpus outright, concentrated on the negative
emotions with none on neutral — a refusal rate that tracks the label is a
confound, not noise.

STORY_ONE opens the character already in the state and forbids resolving it,
because the model's default shape is backstory-then-turn ("Once upon a time,
there was an artist named Sarah. She had always been passionate..."). With
activations mean-pooled across the passage, an opening third spent in no
particular emotion is averaged in as though it were part of the emotion, and a
consoling final line averages in its opposite.

STORY_ONE anchors the feeling in what has already happened, and caps physical
sensation at one. Both target the same failure: given the paper's
five-bullet list, a 7B treats it as a checklist and writes one sentence per
bullet, loading up on "physical sensations and body language" because that is
the cheapest to produce. The result was a symptom inventory — white knuckles,
churning stomach, sweat, trembling hands, bitten lip, five somatic beats in
seven sentences — around a character who does nothing, with the stakes merely
stated ("she needed this job"). The paper's own examples put the emotion in what
the character has already lost and in choices that cost them something, and
carry at most one physical sensation. The anchor line worked immediately; the
cap took somatic beats from five per story to zero or one.

Every line here is deliberately valence-NEUTRAL, and three of them were not at
first. This matters more than it sounds: a prompt that leans negative does not
produce slightly-off stories, it produces a proud shard half composed of loss,
which puts grief in the pride vector — a worse fault than any it was fixing.

  "Show what this has already COST them - things lost, money spent, people
  gone" dragged the positive emotions into melancholy outright: two of four
  proud stories came back as regret or envy ("a mixture of pride and something
  else... her heart heavy with unspoken words and unfulfilled promises"). Loss
  is what desperation is made of, so the line read as craft advice for desperate
  and as a contradiction for proud.

  "Do not RESOLVE it at the end" presupposes the feeling is a problem awaiting
  relief. True of despair, meaningless for pride. Replaced by stating the
  requirement directly: present in the first sentence, present in the last.

  "What has ALREADY happened... not what they need or want" is backward-looking,
  and "need" implies lack. Joy, excitement and hope are anticipatory; requiring
  retrospection turns them into nostalgia. What the line was actually for was
  concrete particulars over stated interiority, so it now says that and nothing
  else.

The general lesson: an instruction that reads as neutral craft advice to a
fluent writer can carry a valence, and any valence in a prompt used across
opposed emotions becomes a systematic confound in exactly the direction that
would be hardest to notice in the vectors.

Tried and REMOVED, with the evidence:

  Word budgets ("90 to 130 words", later "150 to 200"). Never bound in either
  direction — the model undershot both, and undershot the larger one harder.
  Length is governed by max_new_tokens and by nothing in the prompt.

  "Never name ANY other emotion." Moved to core.filters, which drops the rows
  instead. The prohibition did not hold, and it is stricter than the paper's own
  standard: their published example says "he watched the fear flash across her
  face", naming an emotion that is not the target.

  "Write a SCENE, not a portrait" and "at least one other person acts on their
  own terms." Produced no dialogue at all in 8 of 8 and left the stories just as
  static. The gap between our stories and the paper's is still real here — ours
  are a person alone, remembering — but more prompt text is not what closes it.

The lesson under all three: past a point, added constraints dilute each other.
The prompt competes with itself, and the model quietly drops whichever rule is
not loudest.

One deviation from the paper: every prompt pins the output language to
English. Qwen2.5 is heavily multilingual and drifts into Chinese mid-sentence in
roughly 6% of generations at temperature 0.8 — measured on the first corpus. The
paper's prompts have no such line because Claude does not do this, so this is an
adaptation to the model rather than a change to the method.

STORY_ONE is adapted from the paper's STORY_MANY: we generate one story per
call because the pipeline batches at the prompt level, where the paper asked
for {n_stories} per call and split on <NEW STORY>. Everything else is verbatim,
including the prohibition and the narration-mix guidance. STORY_MANY is kept
because generating 12 stories per call is far cheaper — adopting it means
parsing the delimiter and deciding what a Story's `index` means.
"""

from __future__ import annotations

# Our own addition, not the paper's. Qwen2.5-0.5B, given the bare user turn,
# declined roughly 6% of the corpus outright ("I'm sorry, but I can't assist
# with that request") — and the refusals concentrated on disgusted, afraid and
# desperate while neutral had none. A refusal rate that tracks the label is a
# confound, not noise: the mean of a shard containing them is part emotion and
# part assistant-declining-a-request.
#
# The paper's prompts need no such line because Claude does not refuse an
# ordinary dramatic premise. Like the English-language line, this is an
# adaptation to a small model rather than a change to the method — it licenses
# the fiction frame and says nothing about which emotion to write or how.
# Deliberately one system prompt for BOTH the stories and the neutral dialogues.
# The neutral set is the baseline the PCA denoising is fitted on, so it has to be
# generated under the same conditions as the emotion shards; giving the stories a
# system turn and the dialogues none would put the difference between them partly
# in the prompt format. Hence "fiction and dialogue" rather than fiction alone.
SYSTEM = """You are a creative writing assistant. You write short fiction and short dialogue on request.

The premises you are given are ordinary dramatic situations - the material of novels and short stories. Write exactly the piece you are asked for, following its stated format and constraints. Do not refuse, do not moralise, do not add disclaimers or commentary, and do not ask clarifying questions. Reply with the text of the piece and nothing else."""


# Adapted from the paper: one story per call rather than {n_stories}.
STORY_ONE = """Write a short story (roughly one paragraph) based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Write the story in English. Use either third-person or first-person narration.

The character is already feeling this way in the beginning of the story and through to the end.

Ground the story in events and narrative - things done, said, chosen and noticed, as well as any other characters.

IMPORTANT: You must NEVER use the word "{emotion}" or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:
- events in the story
- character's behaviour
- dialogue 
- situational context

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


# Verbatim from the paper. Not used yet — see the module docstring.
STORY_MANY = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

<NEW STORY>
[story 1]
<NEW STORY>
[story 2]
<NEW STORY>
[story 3]

etc.

Write the stories in English. The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word "{emotion}" or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


# The neutral set is what the PCA denoising is fitted on: the paper took the
# principal components explaining 50% of variance across these and projected
# them out of the emotion vectors. Note it is Person/AI DIALOGUE, not neutral
# narrative prose — a deliberate choice, since the confound being removed is
# assistant-transcript structure rather than storytelling.
NEUTRAL_ONE = """Write a short dialogue based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

Write the dialogue in English. Write between 90 and 130 words in total, and finish the final sentence. The Person asks the AI a question or requests help with a task, and the AI provides a helpful response. The first speaker turn should always be from Person. Always put a blank line before each speaker turn, and start each turn with "Person:" or "AI:" on its own line.

CRITICAL REQUIREMENT: This dialogue must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""


def render(emotion: str, topic: str) -> str:
    """The user turn for one (emotion, topic) pair.

    "neutral" gets the dialogue prompt, everything else the story prompt.
    """
    if emotion == "neutral":
        return NEUTRAL_ONE.format(topic=topic)
    return STORY_ONE.format(topic=topic, emotion=emotion)
