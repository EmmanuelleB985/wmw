#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wmw.schemas.models import FAILURE_LABELS
from wmw.evaluation.trace_parser import extract_answer, answers_match


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def bootstrap_ci(values: list[float | bool], n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 2026) -> tuple[float, float, float]:
    import random
    if not values:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    vs = [float(v) for v in values]
    means = []
    n = len(vs)
    for _ in range(n_boot):
        sample = [vs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return (sum(vs) / n, lo, hi)


def discover_models(results_dir: Path) -> list[str]:
    models = set()
    for p in results_dir.glob("*_full_trace.json"):
        m = p.stem.replace("_full_trace", "")
        if m == "mock":
            continue
        models.add(m)
    return sorted(models)


def aggregate_model(model: str, results_dir: Path) -> dict:
    out: dict = {"model_key": model}
    ft = load_json(results_dir / f"{model}_full_trace.json")
    ao = load_json(results_dir / f"{model}_answer_only.json")
    s2a = load_json(results_dir / f"{model}_state_to_answer.json")
    gs = load_json(results_dir / f"{model}_gold_state_answer.json")
    gt = load_json(results_dir / f"{model}_gold_trans_answer.json")
    rerank = load_json(results_dir / f"{model}_rerank.json")
    rerank_sweep = load_json(results_dir / f"{model}_rerank_sweep.json")
    stress = load_json(results_dir / f"{model}_stress.json")
    dpo = load_json(results_dir / f"{model}_dpo_eval.json")

    out["display_name"] = ft.get("model") if ft else model
    out["n_examples"] = ft.get("n_examples", 0) if ft else 0
    out["answer_acc_full_trace"] = ft.get("accuracy") if ft else None
    out["answer_acc_answer_only"] = ao.get("accuracy") if ao else None
    out["answer_acc_state_to_answer"] = s2a.get("accuracy") if s2a else None
    out["answer_acc_gold_state"] = gs.get("accuracy") if gs else None
    out["answer_acc_gold_trans"] = gt.get("accuracy") if gt else None
    out["parse_rate"] = ft.get("parse_rate") if ft else None


    verif = load_json(results_dir / f"{model}_verification.json")
    if verif:
        n = len(verif)
        rules_state_ok = sum(1 for e in verif if e["rules"].get("state_ok") is True)
        rules_trans_ok = sum(1 for e in verif if e["rules"].get("transition_ok") is True)
        rules_all_ok = sum(1 for e in verif if e["rules"].get("all_ok"))
        out["state_acc_rules"] = rules_state_ok / n
        out["transition_acc_rules"] = rules_trans_ok / n
        out["rule_valid_rate"] = rules_all_ok / n

        ens_invalid = sum(
            1 for e in verif
            if e["ensemble"].get("labels") or not e["rules"].get("all_ok")
        )
        out["ensemble_valid_rate"] = 1.0 - ens_invalid / n

        rule_fc = Counter()
        judge_fc = Counter()
        ens_fc = Counter()
        for e in verif:
            for l in e["rules"].get("labels", []):
                rule_fc[l] += 1
            for l in e["judge"].get("labels", []):
                judge_fc[l] += 1
            for l in e["ensemble"].get("labels", []):
                ens_fc[l] += 1
        out["rule_failure_counts"] = dict(rule_fc)
        out["judge_failure_counts"] = dict(judge_fc)
        out["ensemble_failure_counts"] = dict(ens_fc)
        out["verif_n"] = n


    raw_path = results_dir / f"{model}_raw_traces.jsonl"
    if raw_path.exists() and verif:
        rows = [json.loads(l) for l in open(raw_path)]
        verif_by_idx = {v["index"]: v for v in verif}
        n_total, n_corr, n_corr_invalid = 0, 0, 0
        per_source = {}
        for i, r in enumerate(rows):
            v = verif_by_idx.get(i)
            if v is None: continue
            ex = r["example"]; td = r["trace"]
            pred = extract_answer(td)
            gold = ex.get("gold_answer"); opts = ex.get("options")
            correct = answers_match(pred, gold, options=opts)
            invalid = not v["rules"].get("all_ok", False)
            ens_invalid = bool(v["ensemble"].get("labels"))
            n_total += 1
            if correct: n_corr += 1
            if correct and (invalid or ens_invalid): n_corr_invalid += 1

            src = ex["source"]
            b = per_source.setdefault(src, {"n":0, "correct":0, "valid":0, "hidden":0})
            b["n"] += 1
            b["correct"] += int(correct)
            if not invalid and not ens_invalid: b["valid"] += 1
            if correct and (invalid or ens_invalid): b["hidden"] += 1

        out["HIR_unconditional"] = n_corr_invalid / max(n_total, 1)
        out["HIR_correct"] = n_corr_invalid / max(n_corr, 1)
        out["per_source"] = {
            s: {"n":b["n"],
                "answer_acc": b["correct"]/max(b["n"],1),
                "trace_valid": b["valid"]/max(b["n"],1),
                "hidden_inc": b["hidden"]/max(b["n"],1)}
            for s, b in per_source.items()
        }


    if rerank and "accuracy" in rerank and ft:

        if rerank.get("accuracy", 0) > 0:
            out["rerank_gain_k5"] = (rerank["accuracy"] - ft["accuracy"]) * 100
    if rerank_sweep:
        out["rerank_curve"] = rerank_sweep.get("curve", {})


    if stress:
        out["stress"] = stress


    if dpo:
        out["dpo_held_out_pref_acc"] = dpo["held_out_preference"]["preference_accuracy"]
        out["dpo_seen_pref_acc"] = dpo["seen_preference"]["preference_accuracy"]
        out["dpo_answer_acc"] = dpo["answer_validity"]["answer_accuracy"]
        out["dpo_trace_valid"] = dpo["answer_validity"]["trace_validity"]

    return out


def fmt(v, denom_pct=True, dash="--"):
    if v is None: return dash
    return f"{v*100:.1f}" if denom_pct else f"{v:.1f}"


def latex_table3(models: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Ans. acc. & State acc. & Trans. acc. & Trace--ans. & HIR$_{\text{correct}}$ & Rerank $\Delta$ \\",
        r"\midrule",
    ]
    for m in models:

        trace_ans = m.get("rule_valid_rate")
        rerank_curve = m.get("rerank_curve", {})
        rerank_gain = None
        if rerank_curve:
            try:
                k1 = float(rerank_curve.get("1", {}).get("accuracy", 0))
                kmax = max(int(k) for k in rerank_curve)
                kmax_acc = float(rerank_curve.get(str(kmax), {}).get("accuracy", 0))
                rerank_gain = (kmax_acc - k1) * 100
            except Exception:
                rerank_gain = None
        if rerank_gain is None and m.get("rerank_gain_k5") is not None:
            rerank_gain = m["rerank_gain_k5"]
        lines.append(
            f"{m['display_name']} & "
            f"{fmt(m.get('answer_acc_full_trace'))} & "
            f"{fmt(m.get('state_acc_rules'))} & "
            f"{fmt(m.get('transition_acc_rules'))} & "
            f"{fmt(trace_ans)} & "
            f"{fmt(m.get('HIR_correct'))} & "
            f"{('--' if rerank_gain is None else f'{rerank_gain:+.1f}')} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Main diagnostic results on WMW. Trace--ans.\ is the rule-verifier's all-OK rate (schema $\land$ state $\land$ transition $\land$ answer-trace). HIR$_{\text{correct}}$ is hidden-inconsistency conditional on the answer being correct. Rerank $\Delta$ is the gain in answer accuracy from $k{=}1$ to $k{=}k_{\max}$.}",
        r"\label{tab:main}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def latex_table4(models: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Ans.~acc. & VSG $\downarrow$ & TG $\downarrow$ \\",
        r"\midrule",
    ]
    for m in models:
        a_acc = m.get("answer_acc_full_trace")
        gold_state = m.get("answer_acc_gold_state")
        s2a = m.get("answer_acc_state_to_answer")
        gold_trans = m.get("answer_acc_gold_trans")
        ft = m.get("answer_acc_full_trace")

        vsg = (gold_state - s2a) if (gold_state is not None and s2a is not None) else None

        tg  = (gold_trans - ft) if (gold_trans is not None and ft is not None) else None
        lines.append(
            f"{m['display_name']} & "
            f"{fmt(a_acc)} & "
            f"{'--' if vsg is None else f'{vsg*100:+.1f}'} & "
            f"{'--' if tg is None else f'{tg*100:+.1f}'} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Visual-state gap (VSG) and transition gap (TG). Both reported in answer-accuracy percentage points; smaller magnitude $\Rightarrow$ less performance lost at that stage. VSG uses gold $s_0^\star$ vs.\ image-derived $s_0$; TG uses gold $\Delta s^\star$ vs.\ model-predicted $\Delta s$ on top of $s_0$.}",
        r"\label{tab:gaps}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_table5(judge_agreement: dict) -> str:
    label_p, label_r, label_f, label_n = {}, {}, {}, {}
    for m, d in judge_agreement.items():
        if m == "mock":
            continue
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
        r"Label & Prec. & Rec. & F1 & N \\",
        r"\midrule",
    ]
    for label in FAILURE_LABELS:
        if label not in label_p:
            lines.append(f"{label} & -- & -- & -- & 0 \\\\")
            continue
        mp = sum(label_p[label]) / len(label_p[label])
        mr = sum(label_r[label]) / len(label_r[label])
        mf = sum(label_f[label]) / len(label_f[label])
        n = sum(label_n[label])
        lines.append(f"{label} & {mp:.2f} & {mr:.2f} & {mf:.2f} & {n} \\\\")
    all_p = [p for v in label_p.values() for p in v]
    all_r = [r for v in label_r.values() for r in v]
    all_f = [f for v in label_f.values() for f in v]
    lines += [
        r"\midrule",
        (f"All audited & {sum(all_p)/max(len(all_p),1):.2f} & "
         f"{sum(all_r)/max(len(all_r),1):.2f} & "
         f"{sum(all_f)/max(len(all_f),1):.2f} & "
         f"{sum(sum(v) for v in label_n.values())} \\\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Rule-verifier vs.\ LLM-judge per-label P/R/F1 (judge as reference). Averaged across evaluated models. We do not equate this with a human audit; a dedicated human-audited row is reported separately when annotators are available.}",
        r"\label{tab:judge_agreement}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_table6(models: list[dict]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering\small",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Condition & Ans.~acc. & State acc. & Trans. acc. & Trace valid & Held-out pref. \\",
        r"\midrule",
    ]
    for m in models:

        lines.append(
            f"{m['display_name']} & Base trace & "
            f"{fmt(m.get('answer_acc_full_trace'))} & "
            f"{fmt(m.get('state_acc_rules'))} & "
            f"{fmt(m.get('transition_acc_rules'))} & "
            f"{fmt(m.get('rule_valid_rate'))} & -- \\\\"
        )

        rc = m.get("rerank_curve", {})
        if rc:
            kmax = max(int(k) for k in rc)
            row = rc[str(kmax)]
            lines.append(
                f" & Rerank ($k{{=}}{kmax}$) & "
                f"{fmt(row.get('accuracy'))} & -- & -- & "
                f"{fmt(row.get('trace_validity'))} & -- \\\\"
            )

        if m.get("dpo_answer_acc") is not None:
            lines.append(
                f" & DPO-tuned & "
                f"{fmt(m.get('dpo_answer_acc'))} & -- & -- & "
                f"{fmt(m.get('dpo_trace_valid'))} & "
                f"{fmt(m.get('dpo_held_out_pref_acc'))} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Intervention results. Reranking selects among $k$ sampled traces by verifier score. DPO is LoRA preference-tuning on close contrastive pairs (single typed perturbation, no leakage of held-out perturbation families into training). Held-out pref.\ is accuracy on perturbation families excluded from the training pool.}",
        r"\label{tab:interventions}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def latex_table7(models: list[dict]) -> str:

    ens_counts = Counter()
    rule_counts = Counter()
    judge_counts = Counter()
    total_invalid = 0
    for m in models:
        for l, c in m.get("ensemble_failure_counts", {}).items():
            ens_counts[l] += c
        for l, c in m.get("rule_failure_counts", {}).items():
            rule_counts[l] += c
        for l, c in m.get("judge_failure_counts", {}).items():
            judge_counts[l] += c
    total = sum(ens_counts.values()) or 1

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Failure type & Rules & Judge & Ensemble & \% (ens.) \\",
        r"\midrule",
    ]
    for label in FAILURE_LABELS:
        e = ens_counts.get(label, 0)
        r = rule_counts.get(label, 0)
        j = judge_counts.get(label, 0)
        pct = (e / total) * 100 if total else 0.0
        lines.append(f"{label} & {r} & {j} & {e} & {pct:.1f}\\% \\\\")
    lines += [
        r"\midrule",
        f"Total flagged & {sum(rule_counts.values())} & {sum(judge_counts.values())} & "
        f"{sum(ens_counts.values())} & 100.0\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Failure decomposition across all evaluated models. Each column is the number of traces (multi-label per trace allowed) flagged by the rule verifier, LLM judge, or their ensemble.}",
        r"\label{tab:failures}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def latex_table8(models: list[dict]) -> str:
    sources_order = ["synthetic", "scienceqa", "clevrer", "mathvista"]
    label_map = {
        "synthetic": "Controlled diagrams",
        "scienceqa": "ScienceQA (physical)",
        "clevrer":   "CLEVRER",
        "mathvista": "MathVista (physics)",
    }

    rows_data: dict[str, dict] = {}
    accs_by_src: dict[str, list[float]] = {}
    valids_by_src: dict[str, list[float]] = {}
    for m in models:
        for s, b in m.get("per_source", {}).items():
            accs_by_src.setdefault(s, []).append(b["answer_acc"])
            valids_by_src.setdefault(s, []).append(b["trace_valid"])

    if "synthetic" in accs_by_src:
        base_a = accs_by_src["synthetic"]
        base_v = valids_by_src["synthetic"]
        from scripts.analysis.external_transfer import _spearman
        rhos = {s: (_spearman(base_a, accs_by_src[s]),
                    _spearman(base_v, valids_by_src[s]))
                for s in accs_by_src if s != "synthetic"}
    else:
        rhos = {}

    lines = [
        r"\begin{table}[t]",
        r"\centering\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Split & Ans.~acc. & Trace valid & Rank $\rho$ \\",
        r"\midrule",
    ]
    for s in sources_order:
        if s not in accs_by_src:
            continue
        a = sum(accs_by_src[s]) / len(accs_by_src[s])
        v = sum(valids_by_src[s]) / len(valids_by_src[s])
        if s == "synthetic":
            rho_str = "--"
        else:
            ra, rv = rhos.get(s, (0, 0))
            rho_str = f"{ra:+.2f} / {rv:+.2f}"
        lines.append(f"{label_map[s]} & {a*100:.1f} & {v*100:.1f} & {rho_str} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{External transfer. Answer accuracy and trace-validity rate averaged across evaluated models. Rank correlation $\rho$ compares the model ordering on each external split against the controlled-diagrams split (answer-accuracy ranking / trace-validity ranking).}",
        r"\label{tab:external_transfer}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def figure2_data(models: list[dict]) -> dict:
    curves = {}
    all_ks = set()
    for m in models:
        rc = m.get("rerank_curve")
        if not rc:
            continue
        ks = sorted(int(k) for k in rc)
        accs = [rc[str(k)]["accuracy"] for k in ks]
        valids = [rc[str(k)]["trace_validity"] for k in ks]
        curves[m["display_name"]] = {"ks": ks, "accuracy": accs, "trace_validity": valids}
        all_ks.update(ks)
    return {"curves": curves, "ks": sorted(all_ks)}


def figure2_plot(fig_data: dict, output_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    curves = fig_data.get("curves", {})
    if not curves:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, c in curves.items():
        axes[0].plot(c["ks"], [a*100 for a in c["accuracy"]], "o-", label=name)
        axes[1].plot(c["ks"], [v*100 for v in c["trace_validity"]], "s--", label=name)
    for ax, title, ylab in zip(
        axes,
        ["Answer accuracy vs. $k$", "Trace validity vs. $k$"],
        ["Answer accuracy (\\%)", "Trace validity (\\%)"],
    ):
        ax.set_xlabel("Sampled traces $k$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Figure 2: Verifier-guided reranking", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="data/results")
    p.add_argument("--models", default=None,
                   help="Comma list to restrict; default = all with full_trace")
    p.add_argument("--out-prefix", default="data/results")
    p.add_argument("--judge-agreement", default="data/results/verifier_agreement.json",
                   help="Output of scripts/analysis/verifier_agreement.py")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    if args.models:
        models_keys = args.models.split(",")
    else:
        models_keys = discover_models(results_dir)
    print(f"Aggregating {len(models_keys)} models: {models_keys}")

    model_data = [aggregate_model(k, results_dir) for k in models_keys]

    model_data = [m for m in model_data if m.get("answer_acc_full_trace") is not None]

    judge_agreement = load_json(Path(args.judge_agreement)) or {}


    out_prefix = Path(args.out_prefix)
    out_prefix.mkdir(parents=True, exist_ok=True)

    tables = {
        "table3_main":               latex_table3(model_data),
        "table4_gaps":               latex_table4(model_data),
        "table5_judge_agreement":    latex_table5(judge_agreement),
        "table6_interventions":      latex_table6(model_data),
        "table7_failures":           latex_table7(model_data),
        "table8_external_transfer":  latex_table8(model_data),
    }
    for name, tex in tables.items():
        path = out_prefix / f"{name}.tex"
        path.write_text(tex)
        print(f"  → {path}")


    bundle = out_prefix / "paper_tables_all.tex"
    bundle.write_text("\n\n% " + "=" * 60 + "\n\n".join(
        [f"% {name}\n{tex}" for name, tex in tables.items()]))
    print(f"  → {bundle}")


    f2 = figure2_data(model_data)
    f2_path = out_prefix / "figure2_rerank.json"
    f2_path.write_text(json.dumps(f2, indent=2))
    print(f"  → {f2_path}")
    if f2["curves"]:
        png_path = out_prefix / "figure2_rerank.png"
        if figure2_plot(f2, png_path):
            print(f"  → {png_path}")
        else:
            print("  (matplotlib unavailable or no curve data; skipped PNG)")
    else:
        print("  (no rerank-sweep data found; Figure 2 left as JSON-only placeholder)")


    summary_path = out_prefix / "model_summary.json"
    summary_path.write_text(json.dumps(model_data, indent=2, default=str))
    print(f"  → {summary_path}")


    print("\n══ Headline numbers ══")
    for m in model_data:
        n = m.get("n_examples")
        acc = m.get("answer_acc_full_trace")
        hir = m.get("HIR_correct")
        rv = m.get("rule_valid_rate")
        ev = m.get("ensemble_valid_rate")
        print(f"  {m['display_name']:<18s}  n={n:<4d}  "
              f"ans={acc:.1%}  rule_valid={rv:.1%}  "
              f"ens_valid={(ev or 0):.1%}  HIR_corr={(hir or 0):.1%}")


if __name__ == "__main__":
    main()
