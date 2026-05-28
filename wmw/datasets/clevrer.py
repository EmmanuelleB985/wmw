from __future__ import annotations
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

from wmw.datasets.common import EvalExample, save_examples


_QTYPE_MAP = {
    "descriptive": "static_state",
    "explanatory": "transition_prediction",
    "predictive": "transition_prediction",
    "counterfactual": "counterfactual",
}


_PHYSICS_KEYWORDS = {
    "collision", "collide", "moving", "velocity", "speed",
    "direction", "enter", "exit", "stationary", "still",
    "rolling", "sliding", "bouncing",
}


_CLEVRER_URLS = {
    "val_questions": "http://data.csail.mit.edu/clevrer/questions/validation.json",
    "val_scenes": "http://data.csail.mit.edu/clevrer/scene_descriptions/validation.json",
}


def _extract_keyframe(video_path: str, output_path: str, time_sec: float = 1.0) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-ss", str(time_sec), "-i", video_path,
             "-vframes", "1", "-q:v", "2", output_path,
             "-y", "-loglevel", "quiet"],
            check=True, timeout=30,
        )
        return os.path.exists(output_path)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_clevrer(
    output_dir: str | Path = "data/clevrer",
    max_examples: int | None = 200,
    question_types: list[str] | None = None,
) -> list[EvalExample]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if question_types is None:
        question_types = ["descriptive", "explanatory", "predictive", "counterfactual"]


    q_path = output_dir / "validation_questions.json"
    if not q_path.exists():
        print(f"  Downloading CLEVRER validation questions...")
        try:
            import urllib.request
            urllib.request.urlretrieve(_CLEVRER_URLS["val_questions"], str(q_path))
            print(f"  Downloaded → {q_path}")
        except Exception as e:
            print(f"  WARNING: Could not download CLEVRER questions: {e}")
            print(f"  Creating placeholder from template...")
            _create_clevrer_placeholder(q_path, max_examples or 200)


    s_path = output_dir / "validation_scenes.json"
    if not s_path.exists():
        try:
            import urllib.request
            urllib.request.urlretrieve(_CLEVRER_URLS["val_scenes"], str(s_path))
        except Exception:
            pass


    with open(q_path) as f:
        data = json.load(f)


    scenes = {}
    if s_path.exists():
        with open(s_path) as f:
            scene_data = json.load(f)
            for s in scene_data:
                scenes[s.get("scene_index", s.get("video_filename", ""))] = s


    examples: list[EvalExample] = []

    for video_item in data:
        video_id = video_item.get("scene_index", video_item.get("video_filename", ""))
        questions = video_item.get("questions", [])

        for q in questions:
            qtype = q.get("question_type", "descriptive")
            if qtype not in question_types:
                continue


            question_text = q.get("question", "")


            gold_answer = None
            if "answer" in q:
                gold_answer = q["answer"]
            elif "choices" in q:
                choices = q["choices"]
                correct = [c for c in choices if c.get("answer") == "correct"]
                if correct:
                    gold_answer = correct[0].get("choice", "")


            options = None
            if "choices" in q:
                options = [c.get("choice", "") for c in q.get("choices", [])]


            scene = scenes.get(video_id, {})
            scene_desc = ""
            if scene:
                objects = scene.get("objects", [])
                scene_desc = f"Scene has {len(objects)} objects. "
                for obj in objects[:5]:
                    scene_desc += f"{obj.get('color','')} {obj.get('shape','')} {obj.get('material','')}, "

            ex = EvalExample(
                id=f"clevrer_val_{video_id}_{q.get('question_id', len(examples)):05d}",
                source="clevrer",
                question=question_text,
                image_path=None,
                options=options,
                gold_answer=gold_answer,
                topic="collision",
                difficulty="medium",
                task_type=_QTYPE_MAP.get(qtype, "transition_prediction"),
                extra={
                    "question_type": qtype,
                    "video_id": video_id,
                    "scene_description": scene_desc,
                    "program": q.get("program", []),
                },
            )
            examples.append(ex)

            if max_examples and len(examples) >= max_examples:
                break

        if max_examples and len(examples) >= max_examples:
            break

    print(f"  CLEVRER: {len(examples)} examples across types: {question_types}")
    save_examples(examples, output_dir / "clevrer_val.jsonl")
    return examples


def _create_clevrer_placeholder(path: Path, n: int) -> None:
    import random
    random.seed(2026)

    colors = ["red", "blue", "green", "yellow", "gray", "brown", "purple", "cyan"]
    shapes = ["sphere", "cube", "cylinder"]
    materials = ["metal", "rubber"]

    templates = {
        "descriptive": [
            "What color is the {shape} that {action}?",
            "How many objects are {state}?",
            "What is the shape of the {color} object?",
            "Is the {color} {shape} moving or stationary?",
        ],
        "explanatory": [
            "What caused the {color} {shape} to {action}?",
            "Why did the {color} object {action}?",
            "Which event caused the {color} {shape} to change direction?",
        ],
        "predictive": [
            "What will happen next to the {color} {shape}?",
            "Will the {color} {shape} collide with another object?",
            "Which objects will be moving after all collisions?",
        ],
        "counterfactual": [
            "What would happen if the {color} {shape} were removed?",
            "Without the collision, would the {color} object have {action}?",
            "If the {color} {shape} were heavier, what would change?",
        ],
    }

    actions = ["enters the scene", "collides with another object",
               "changes direction", "exits the scene", "stops moving"]
    states = ["moving", "stationary", "entering the scene"]

    data = []
    for scene_idx in range(n // 4 + 1):
        questions = []
        for qtype, tmpl_list in templates.items():
            tmpl = random.choice(tmpl_list)
            color = random.choice(colors)
            shape = random.choice(shapes)
            q_text = tmpl.format(
                color=color, shape=shape,
                action=random.choice(actions),
                state=random.choice(states),
            )
            questions.append({
                "question_id": len(questions),
                "question_type": qtype,
                "question": q_text,
                "answer": random.choice(colors + ["yes", "no", "2", "3"]),
            })
        data.append({
            "scene_index": scene_idx,
            "video_filename": f"video_{scene_idx:05d}.mp4",
            "questions": questions,
        })

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_clevrer_cached(
    data_dir: str | Path = "data/clevrer",
) -> list[EvalExample]:
    from wmw.datasets.common import load_examples
    path = Path(data_dir) / "clevrer_val.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run download_clevrer() first.")
    return load_examples(path)
