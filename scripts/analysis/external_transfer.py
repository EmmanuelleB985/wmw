#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wmw.evaluation.trace_parser import extract_answer, answers_match


def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2 or len(b) != n:
        return 0.0

    def ranks(xs):
        sorted_idx = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[sorted_idx[j + 1]] == xs[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[sorted_idx[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    den_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)


def per_source_metrics(model_name: str, results_dir: Path) -> dict:
    raw_path = results_dir / f"{model_name}_raw_traces.jsonl"
    verif_path = results_dir / f"{model_name}_verification.json"
    if not raw_path.exists():
        return {}

    rows = []
    with open(raw_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)


    verif = {}
    if verif_path.exists():
        for v in json.load(open(verif_path)):
            verif[v["index"]] = v

    by_source: dict[str, dict] = {}
    for i, row in enumerate(rows):
        src = row["example"]["source"]
        ex = row["example"]
        td = row["trace"]
        pred = extract_answer(td)
        gold = ex.get("gold_answer")
        opts = ex.get("options")
        correct = answers_match(pred, gold, options=opts) if pred is not None else False
        v = verif.get(i, {})
        v_rules = v.get("rules", {})
        v_judge = v.get("judge", {})
        v_ens = v.get("ensemble", {})
        bucket = by_source.setdefault(src, {
            "n": 0, "correct": 0, "state_ok": 0, "transition_ok": 0,
            "ensemble_valid": 0, "rule_valid": 0,
            "hidden_inconsistency": 0,
        })
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        if v_rules.get("state_ok") is True:
            bucket["state_ok"] += 1
        if v_rules.get("transition_ok") is True:
            bucket["transition_ok"] += 1
        rule_valid = (v_rules.get("state_ok") is True and
                      v_rules.get("transition_ok") is True and
                      v_rules.get("all_ok", False))
        if rule_valid:
            bucket["rule_valid"] += 1

        ens_labels = v_ens.get("labels", [])
        ens_valid = rule_valid and not ens_labels
        if ens_valid:
            bucket["ensemble_valid"] += 1

        if correct and not rule_valid:
            bucket["hidden_inconsistency"] += 1

    metrics = {}
    for src, b in by_source.items():
        n = b["n"]
        metrics[src] = {
            "n": n,
            "answer_acc": b["correct"] / n if n else 0.0,
            "rule_valid_rate": b["rule_valid"] / n if n else 0.0,
            "ensemble_valid_rate": b["ensemble_valid"] / n if n else 0.0,
            "state_acc": b["state_ok"] / n if n else 0.0,
            "transition_acc": b["transition_ok"] / n if n else 0.0,
            "hidden_inconsistency": b["hidden_inconsistency"] / n if n else 0.0,
        }
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--models", default=None,
                   help="Comma list of model keys to include; default = all with raw_traces")
    p.add_argument("--output", default="data/results/external_transfer.json")
    p.add_argument("--tex-output", default="data/results/table_external_transfer.tex")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if args.models:
        models = args.models.split(",")
    else:
        models = sorted({p.stem.replace("_raw_traces", "")
                         for p in results_dir.glob("*_raw_traces.jsonl")})
    print(f"Models: {models}")

    by_model: dict[str, dict] = {}
    for m in models:
        by_model[m] = per_source_metrics(m, results_dir)


    sources_order = ["synthetic", "scienceqa", "clevrer", "mathvista"]
    sources_present = sorted({s for d in by_model.values() for s in d.keys()},
                              key=lambda s: sources_order.index(s) if s in sources_order else 99)

    per_src_accs = {
        s: [by_model[m].get(s, {}).get("answer_acc", 0.0) for m in models]
        for s in sources_present
    }
    per_src_valid = {
        s: [by_model[m].get(s, {}).get("rule_valid_rate", 0.0) for m in models]
        for s in sources_present
    }
    rank_corrs = {}
    if "synthetic" in per_src_accs:
        base = per_src_accs["synthetic"]
        for s in sources_present:
            if s == "synthetic":
                continue
            rank_corrs[s] = {
                "spearman_answer_acc": round(_spearman(base, per_src_accs[s]), 3),
                "spearman_rule_valid": round(
                    _spearman(per_src_valid["synthetic"], per_src_valid[s]), 3),
            }

    out = {
        "models": models,
        "by_model": by_model,
        "rank_corrs": rank_corrs,
        "sources_present": sources_present,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {args.output}")

    print("\nPer-model accuracy by source:")
    for m in models:
        line = f"  {m:<14s} "
        for s in sources_present:
            d = by_model[m].get(s, {})
            line += f" {s}={d.get('answer_acc', 0):.1%}"
        print(line)
    print("\nSpearman rank-corr vs synthetic:")
    for s, d in rank_corrs.items():
        print(f"  {s:<12s} answer_acc ρ={d['spearman_answer_acc']:+.3f}  "
              f"rule_valid ρ={d['spearman_rule_valid']:+.3f}")


    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Split & Answer acc. & Trace valid. & Rank $\rho$ (vs.~synth) \\",
        r"\midrule",
    ]

    def mean(xs): return sum(xs) / max(len(xs), 1)
    label_map = {
        "synthetic": "Controlled diagrams",
        "scienceqa": "ScienceQA (physical)",
        "clevrer":   "CLEVRER",
        "mathvista": "MathVista (physics)",
    }
    for s in sources_present:
        accs = [by_model[m].get(s, {}).get("answer_acc", None) for m in models]
        valids = [by_model[m].get(s, {}).get("rule_valid_rate", None) for m in models]
        accs = [a for a in accs if a is not None]
        valids = [v for v in valids if v is not None]
        if not accs:
            continue
        rho_a = rank_corrs.get(s, {}).get("spearman_answer_acc")
        rho_v = rank_corrs.get(s, {}).get("spearman_rule_valid")
        rho_str = "--" if s == "synthetic" else f"{rho_a:+.2f} / {rho_v:+.2f}"
        lines.append(
            f"{label_map.get(s, s)} & "
            f"{mean(accs)*100:.1f}\\% & "
            f"{mean(valids)*100:.1f}\\% & "
            f"{rho_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{External transfer of WMW diagnostics. Answer accuracy and "
        r"trace-validity rate averaged across evaluated models. Rank "
        r"correlation $\rho$ compares model orderings on each external split "
        r"against the controlled-diagrams split (answer-accuracy ranking / "
        r"rule-valid ranking).}",
        r"\label{tab:external_transfer}",
        r"\end{table}",
    ]
    Path(args.tex_output).write_text("\n".join(lines))
    print(f"  → {args.tex_output}")


if __name__ == "__main__":
    main()
