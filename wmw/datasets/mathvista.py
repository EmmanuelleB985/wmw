from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

from wmw.datasets.common import EvalExample, save_examples


_PHYSICS_KEYWORDS = {
    "force", "velocity", "acceleration", "momentum", "energy",
    "friction", "gravity", "mass", "weight", "spring",
    "incline", "pulley", "circuit", "resistor", "voltage",
    "current", "pressure", "buoyancy", "density", "wave",
    "frequency", "wavelength", "optics", "lens", "mirror",
    "pendulum", "oscillation", "torque", "lever", "collision",
    "projectile", "free fall", "newton", "joule", "watt",
    "ampere", "ohm", "pascal", "hertz", "temperature",
    "heat", "thermodynamic", "kinetic", "potential",
    "displacement", "distance", "speed", "trajectory",
    "equilibrium", "tension", "normal force", "centripetal",
    "angular", "rotation", "conservation",

    "free body diagram", "circuit diagram", "ray diagram",
    "force diagram", "vector", "slope", "ramp",
}


_PHYSICS_CATEGORIES = {
    "physics", "science", "mechanics", "thermodynamics",
    "electromagnetism", "optics", "waves",
}


def _is_physics_mathvista(item: dict) -> bool:

    question = str(item.get("question", "")).lower()
    for kw in _PHYSICS_KEYWORDS:
        if kw in question:
            return True


    metadata = item.get("metadata", {})
    if isinstance(metadata, dict):
        category = str(metadata.get("category", "")).lower()
        subject = str(metadata.get("subject", "")).lower()
        context = str(metadata.get("context", "")).lower()
        for kw in _PHYSICS_CATEGORIES:
            if kw in category or kw in subject or kw in context:
                return True


    query = str(item.get("query", "")).lower()
    for kw in _PHYSICS_KEYWORDS:
        if kw in query:
            return True

    return False


def download_mathvista(
    output_dir: str | Path = "data/mathvista",
    split: str = "testmini",
    max_examples: int | None = None,
    physics_only: bool = True,
) -> list[EvalExample]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install 'datasets' package: pip install datasets Pillow")
        sys.exit(1)

    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading MathVista ({split} split)...")
    ds = load_dataset("AI4Math/MathVista", split=split, trust_remote_code=True)
    print(f"  Raw examples: {len(ds)}")

    examples: list[EvalExample] = []
    skipped = 0

    for idx, item in enumerate(ds):
        if physics_only and not _is_physics_mathvista(item):
            skipped += 1
            continue


        img_path = None
        image_field = item.get("decoded_image") or item.get("image")
        if image_field is not None:
            try:
                img_filename = f"mathvista_{split}_{idx:05d}.png"
                img_path_full = img_dir / img_filename
                if hasattr(image_field, "save"):
                    image_field.save(str(img_path_full))
                    img_path = str(img_path_full)
            except Exception:
                img_path = None


        question = item.get("question", item.get("query", ""))
        gold_answer = item.get("answer", None)


        options = None
        choices = item.get("choices", [])
        if choices and isinstance(choices, list):
            options = choices


        metadata = item.get("metadata", {})
        difficulty = "medium"
        if isinstance(metadata, dict):
            grade = str(metadata.get("grade", "")).lower()
            if "elementary" in grade or "1" in grade or "2" in grade:
                difficulty = "easy"
            elif "college" in grade or "graduate" in grade:
                difficulty = "hard"

        ex = EvalExample(
            id=f"mathvista_{split}_{idx:05d}",
            source="mathvista",
            question=question,
            image_path=img_path,
            options=options,
            gold_answer=gold_answer,
            topic=str(metadata.get("category", "")) if isinstance(metadata, dict) else "",
            difficulty=difficulty,
            task_type="answer_qa",
            extra={
                "original_index": idx,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "pid": item.get("pid", ""),
            },
        )
        examples.append(ex)

        if max_examples and len(examples) >= max_examples:
            break

    print(f"  Physics filter: skipped {skipped} non-physics")
    print(f"  Final: {len(examples)} examples")

    save_examples(examples, output_dir / f"mathvista_{split}.jsonl")
    return examples


def load_mathvista_cached(
    data_dir: str | Path = "data/mathvista",
    split: str = "testmini",
) -> list[EvalExample]:
    from wmw.datasets.common import load_examples
    path = Path(data_dir) / f"mathvista_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_mathvista() first.")
    return load_examples(path)
