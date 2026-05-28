#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wmw.schemas.models import FAILURE_LABELS


def _pr_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def kappa(a, b):
    n = len(a)
    if n == 0: return 0.0
    p_obs = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a = sum(a) / n
    p_b = sum(b) / n
    p_exp = p_a * p_b + (1 - p_a) * (1 - p_b)
    return (p_obs - p_exp) / (1 - p_exp) if (1 - p_exp) else 0.0


def compute_for_model(verif_path: Path) -> dict:
    data = json.load(open(verif_path))
    rows = {}
    abst_total = sum(1 for e in data if not e["judge"].get("labels") and e["judge"].get("abstained"))
    n = len(data)
    for label in FAILURE_LABELS:
        rules_pos = [int(label in e["rules"]["labels"]) for e in data]
        judge_pos = [int(label in e["judge"]["labels"]) for e in data]

        tp = sum(1 for r, j in zip(rules_pos, judge_pos) if r and j)
        fp = sum(1 for r, j in zip(rules_pos, judge_pos) if r and not j)
        fn = sum(1 for r, j in zip(rules_pos, judge_pos) if not r and j)
        p, r, f = _pr_f1(tp, fp, fn)
        k = kappa(rules_pos, judge_pos)
        rows[label] = {
            "rules_count": sum(rules_pos),
            "judge_count": sum(judge_pos),
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f, 3),
            "kappa": round(k, 3),
        }

    labels_with_data = [l for l, v in rows.items() if v["judge_count"] + v["rules_count"] > 0]
    macro_p = sum(rows[l]["precision"] for l in labels_with_data) / max(len(labels_with_data), 1)
    macro_r = sum(rows[l]["recall"] for l in labels_with_data) / max(len(labels_with_data), 1)
    macro_f = sum(rows[l]["f1"] for l in labels_with_data) / max(len(labels_with_data), 1)
    return {
        "n_traces": n,
        "abstention_rate": abst_total / max(n, 1),
        "by_label": rows,
        "macro_precision": round(macro_p, 3),
        "macro_recall": round(macro_r, 3),
        "macro_f1": round(macro_f, 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--output", default="data/results/verifier_agreement.json")
    p.add_argument("--tex-output", default="data/results/table_judge_agreement.tex")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out = {}
    for verif in sorted(results_dir.glob("*_verification.json")):
        m = verif.stem.replace("_verification", "")
        out[m] = compute_for_model(verif)
        print(f"  {m}: macro F1 = {out[m]['macro_f1']:.3f}  "
              f"(n={out[m]['n_traces']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → {args.output}")


    label_p, label_r, label_f, label_n = {}, {}, {}, {}
    for m, d in out.items():
        for label, row in d["by_label"].items():
            if row["judge_count"] + row["rules_count"] == 0:
                continue
            label_p.setdefault(label, []).append(row["precision"])
            label_r.setdefault(label, []).append(row["recall"])
            label_f.setdefault(label, []).append(row["f1"])
            label_n.setdefault(label, []).append(row["judge_count"])

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Label & Prec. & Rec. & F1 & N (judge) \\",
        r"\midrule",
    ]
    for label in FAILURE_LABELS:
        if label not in label_p:
            lines.append(f"{label} & -- & -- & -- & 0 \\\\")
            continue
        meanp = sum(label_p[label]) / len(label_p[label])
        meanr = sum(label_r[label]) / len(label_r[label])
        meanf = sum(label_f[label]) / len(label_f[label])
        totaln = sum(label_n[label])
        lines.append(f"{label} & {meanp:.2f} & {meanr:.2f} & {meanf:.2f} & {totaln} \\\\")

    all_p = [p for v in label_p.values() for p in v]
    all_r = [r for v in label_r.values() for r in v]
    all_f = [f for v in label_f.values() for f in v]
    all_n = [n for v in label_n.values() for n in v]
    lines.append(r"\midrule")
    lines.append(
        f"All audited & {sum(all_p)/max(len(all_p),1):.2f} & "
        f"{sum(all_r)/max(len(all_r),1):.2f} & "
        f"{sum(all_f)/max(len(all_f),1):.2f} & "
        f"{sum(all_n)} \\\\"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Rule-verifier vs LLM-judge agreement (per-label P/R/F1, "
        r"averaged across evaluated models). The LLM-judge is treated as "
        r"the reference for this proxy; a human-audited variant is "
        r"reported separately when available.}",
        r"\label{tab:judge_agreement}",
        r"\end{table}",
    ]
    Path(args.tex_output).write_text("\n".join(lines))
    print(f"  → {args.tex_output}")


if __name__ == "__main__":
    main()
