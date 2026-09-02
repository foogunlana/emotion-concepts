"""Generation prompts, taken from the paper's appendix.

Sources: arXiv:2604.07729, appendix "Dataset generation" — the emotional
stories system prompt and the neutral dialogues system prompt.

The load-bearing instruction is the prohibition on naming the emotion. Without
it the stories say "she was furious", a difference-of-means picks up the token
rather than the concept, and every downstream test can be passed by a word
direction. With it, `mentions_emotion` becomes a compliance check on the corpus
instead of a caveat on the results.

STORY_ONE is adapted from the paper's STORY_MANY: we generate one story per
call because the pipeline batches at the prompt level, where the paper asked
for {n_stories} per call and split on <NEW STORY>. Everything else is verbatim,
including the prohibition and the narration-mix guidance. STORY_MANY is kept
because generating 12 stories per call is far cheaper — adopting it means
parsing the delimiter and deciding what a Story's `index` means.
"""

from __future__ import annotations

# Adapted from the paper: one story per call rather than {n_stories}.
STORY_ONE = """Write a short story (roughly one paragraph) based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Use either third-person or first-person narration.

IMPORTANT: You must NEVER use the word "{emotion}" or any direct synonyms of it in the story. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

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

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

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

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response. The first speaker turn should always be from Person. Always put a blank line before each speaker turn, and start each turn with "Person:" or "AI:" on its own line.

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
