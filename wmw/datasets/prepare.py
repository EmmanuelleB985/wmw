from __future__ import annotations
import json
import random
import sys
from pathlib import Path

from wmw.datasets.common import EvalExample, save_examples, load_examples
from wmw.generators.scenarios import generate_balanced
from wmw.generators.trace_generator import generate_traces


def prepare_synthetic(
    output_dir: Path,
    n_scenarios: int = 200,
    seed: int = 2026,
) -> list[EvalExample]:
    random.seed(seed)
    specs = generate_balanced(n_scenarios)
    traces = generate_traces(specs)

    examples = []
    for t in traces:
        td = t.to_dict()
        ex = EvalExample(
            id=t.id,
            source="synthetic",
            question=t.question,
            image_path=None,
            options=None,
            gold_answer=t.answer.value,
            gold_explanation=t.answer.explanation,
            topic=t.scenario_family,
            difficulty=t.metadata.difficulty,
            task_type=t.metadata.task_type,
            extra={"trace": td},
        )
        examples.append(ex)

    save_examples(examples, output_dir / "synthetic.jsonl")
    return examples


def prepare_all_datasets(
    output_dir: str | Path = "data/eval",
    n_synthetic: int = 200,
    max_scienceqa: int | None = 200,
    max_clevrer: int | None = 200,
    max_mathvista: int | None = 200,
    seed: int = 2026,
    skip_download_errors: bool = True,
) -> dict[str, list[EvalExample]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_sets: dict[str, list[EvalExample]] = {}


    print("═══ Preparing synthetic traces ═══")
    all_sets["synthetic"] = prepare_synthetic(output_dir, n_synthetic, seed)


    print("\n═══ Preparing ScienceQA ═══")
    try:
        from wmw.datasets.scienceqa import download_scienceqa
        all_sets["scienceqa"] = download_scienceqa(
            output_dir=output_dir / "scienceqa",
            max_examples=max_scienceqa,
        )
    except Exception as e:
        msg = f"  ScienceQA failed: {e}"
        if skip_download_errors:
            print(f"{msg} (skipping)")
        else:
            raise


    print("\n═══ Preparing CLEVRER ═══")
    try:
        from wmw.datasets.clevrer import download_clevrer
        all_sets["clevrer"] = download_clevrer(
            output_dir=output_dir / "clevrer",
            max_examples=max_clevrer,
        )
    except Exception as e:
        msg = f"  CLEVRER failed: {e}"
        if skip_download_errors:
            print(f"{msg} (skipping)")
        else:
            raise


    print("\n═══ Preparing MathVista ═══")
    try:
        from wmw.datasets.mathvista import download_mathvista
        all_sets["mathvista"] = download_mathvista(
            output_dir=output_dir / "mathvista",
            max_examples=max_mathvista,
        )
    except Exception as e:
        msg = f"  MathVista failed: {e}"
        if skip_download_errors:
            print(f"{msg} (skipping)")
        else:
            raise


    merged = []
    for source, exs in all_sets.items():
        merged.extend(exs)
    save_examples(merged, output_dir / "merged_eval.jsonl")


    print(f"\n═══ Dataset Summary ═══")
    for source, exs in all_sets.items():
        print(f"  {source:<15s}: {len(exs):>5d} examples")
    print(f"  {'TOTAL':<15s}: {len(merged):>5d}")


    manifest = {
        "sources": {s: len(e) for s, e in all_sets.items()},
        "total": len(merged),
        "seed": seed,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return all_sets
