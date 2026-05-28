from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import json


@dataclass
class EvalExample:
    id: str
    source: str
    question: str
    image_path: str | None = None
    image_url: str | None = None
    options: list[str] | None = None
    gold_answer: str | float | None = None
    gold_explanation: str | None = None
    topic: str = ""
    difficulty: str = "medium"
    task_type: str = "transition_prediction"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "question": self.question,
            "image_path": self.image_path,
            "image_url": self.image_url,
            "options": self.options,
            "gold_answer": self.gold_answer,
            "gold_explanation": self.gold_explanation,
            "topic": self.topic,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "extra": self.extra,
        }


def save_examples(examples: list[EvalExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            json.dump(ex.to_dict(), f)
            f.write("\n")
    print(f"  Saved {len(examples)} examples → {path}")


def load_examples(path: Path) -> list[EvalExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                examples.append(EvalExample(**d))
    return examples
