from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any

from wmw.datasets.common import EvalExample, save_examples


_PHYSICS_TOPICS = {
    "force and motion", "forces", "friction", "gravity", "velocity",
    "acceleration", "energy", "momentum", "work", "power",
    "simple machines", "pulleys", "levers", "inclined planes",
    "waves", "sound", "light", "optics", "electricity",
    "magnetism", "circuits", "heat", "temperature", "pressure",
    "buoyancy", "density", "mass", "weight", "newton",
    "kinetic energy", "potential energy", "conservation",
    "thermal energy", "heat transfer", "conduction", "convection",
    "radiation", "reflection", "refraction",

    "physical sciences", "physics",
}


_PHYSICS_SUBJECTS = {"natural science"}
_PHYSICS_CATEGORIES = {"Physics", "physical science"}


def _is_physics(example: dict) -> bool:
    topic = str(example.get("topic", "")).lower()
    subject = str(example.get("subject", "")).lower()
    category = str(example.get("category", "")).lower()
    hint = str(example.get("hint", "")).lower()
    question = str(example.get("question", "")).lower()


    for kw in _PHYSICS_TOPICS:
        if kw in topic or kw in category or kw in hint or kw in question:
            return True


    if subject in _PHYSICS_SUBJECTS and any(
        kw in topic or kw in category for kw in _PHYSICS_CATEGORIES
    ):
        return True

    return False


def _has_image(example: dict) -> bool:
    return example.get("image") is not None and str(example.get("image", "")).strip() != ""


def download_scienceqa(
    output_dir: str | Path = "data/scienceqa",
    split: str = "test",
    max_examples: int | None = None,
    physics_only: bool = True,
    require_image: bool = True,
) -> list[EvalExample]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install 'datasets' package: pip install datasets Pillow")
        sys.exit(1)

    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading ScienceQA ({split} split)...")
    ds = load_dataset("derek-thomas/ScienceQA", split=split, trust_remote_code=True)
    print(f"  Raw examples: {len(ds)}")

    examples: list[EvalExample] = []
    skipped_no_physics = 0
    skipped_no_image = 0

    for idx, item in enumerate(ds):
        if physics_only and not _is_physics(item):
            skipped_no_physics += 1
            continue

        if require_image and not _has_image(item):
            skipped_no_image += 1
            continue


        img_path = None
        if item.get("image") is not None:
            try:
                img = item["image"]
                img_filename = f"scienceqa_{split}_{idx:05d}.png"
                img_path_full = img_dir / img_filename
                if hasattr(img, "save"):
                    img.save(str(img_path_full))
                    img_path = str(img_path_full)
            except Exception as e:
                img_path = None


        choices = item.get("choices", [])
        answer_idx = item.get("answer")
        gold_answer = None
        if isinstance(answer_idx, int) and answer_idx < len(choices):
            gold_answer = choices[answer_idx]

        ex = EvalExample(
            id=f"scienceqa_{split}_{idx:05d}",
            source="scienceqa",
            question=item.get("question", ""),
            image_path=img_path,
            options=choices if choices else None,
            gold_answer=gold_answer,
            gold_explanation=item.get("solution", None),
            topic=str(item.get("topic", "")),
            difficulty="medium",
            task_type="answer_qa",
            extra={
                "subject": item.get("subject", ""),
                "category": item.get("category", ""),
                "hint": item.get("hint", ""),
                "original_index": idx,
                "answer_index": answer_idx,
            },
        )
        examples.append(ex)

        if max_examples and len(examples) >= max_examples:
            break

    print(f"  Physics filter: skipped {skipped_no_physics} non-physics")
    print(f"  Image filter: skipped {skipped_no_image} without images")
    print(f"  Final: {len(examples)} examples")

    save_examples(examples, output_dir / f"scienceqa_{split}.jsonl")
    return examples


def load_scienceqa_cached(
    data_dir: str | Path = "data/scienceqa",
    split: str = "test",
) -> list[EvalExample]:
    from wmw.datasets.common import load_examples
    path = Path(data_dir) / f"scienceqa_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_scienceqa() first.")
    return load_examples(path)
