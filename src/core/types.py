from dataclasses import dataclass

@dataclass
class Prompt:
    emotion: str
    topic: str
    index: int
    instruction: str

@dataclass
class Story:
    emotion: str
    topic: str
    index: int
    prompt: str
    text: str
    seed: int
    mentions_emotion: bool