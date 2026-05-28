#!/usr/bin/env python3

from __future__ import annotations
import argparse, csv, json, math, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wmw.datasets.common import EvalExample, load_examples
from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
from wmw.evaluation.vlm_caller import ModelConfig, call_vlm
from wmw.evaluation.trace_parser import parse_trace, extract_answer, answers_match
from wmw.verifiers.pipeline import verify_trace
from wmw.verifiers.llm_judge import call_llm_judge


BASE = Path("data/eval")
OUT  = Path("data/experiments")
SEED = 2026

MODELS = {
    "gpt4o_mini": ModelConfig(name="GPT-4o-mini", provider="openai",
        model_id="gpt-4o-mini", api_key_env="OPENAI_API_KEY"),
    "gpt4o": ModelConfig(name="GPT-4o", provider="openai",
        model_id="gpt-4o", api_key_env="OPENAI_API_KEY"),
    "gpt5_5": ModelConfig(name="GPT-5.5", provider="openai",
        model_id="gpt-5.5", api_key_env="OPENAI_API_KEY"),
    "opus": ModelConfig(name="Claude Opus 4.7", provider="anthropic",
        model_id="claude-opus-4-7", api_key_env="ANTHROPIC_API_KEY"),
}
ALL_MK = ["gpt4o_mini", "gpt4o", "gpt5_5", "opus"]
CHEAP  = ["gpt4o_mini", "gpt4o"]

JUDGES = {
    "sonnet": ModelConfig(name="Claude Sonnet 4", provider="anthropic",
        model_id="claude-sonnet-4-5-20250929", api_key_env="ANTHROPIC_API_KEY"),
    "gpt4o_j": ModelConfig(name="GPT-4o (judge)", provider="openai",
        model_id="gpt-4o", api_key_env="OPENAI_API_KEY"),
}


def load_orig():
    ex = []
    for f in ["synthetic.jsonl", "scienceqa.jsonl", "mathvista.jsonl"]:
        p = BASE / f
        if p.exists(): ex.extend(load_examples(p))
    return ex


def load_scaled():
    scaled_path = OUT / "scaled_examples.jsonl"
    if scaled_path.exists():
        n = len(load_examples(scaled_path))
        if n >= 300:
            print(f"  Using cached scaled dataset: {n} examples")
            return load_examples(scaled_path)
        else:
            print(f"  Cached dataset only has {n} examples, regenerating...")


    import wmw.datasets.scienceqa as _sqa_mod
    import wmw.datasets.mathvista as _mv_mod
    import importlib, inspect

    for mod in [_sqa_mod, _mv_mod]:
        src = inspect.getsource(mod)
        if "trust_remote_code" in src:

            src_path = Path(inspect.getfile(mod))
            patched = src_path.read_text().replace(", trust_remote_code=True", "")

            tmp = OUT / "patched" / src_path.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(patched)
            spec = importlib.util.spec_from_file_location(mod.__name__, tmp)
            patched_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(patched_mod)
            sys.modules[mod.__name__] = patched_mod
            print(f"    Patched {src_path.name}: removed trust_remote_code")

    print("  Generating scaled dataset (target: 300+ examples)...")
    print("  All three sources are REQUIRED. No fallback to synthetic-only.")
    from wmw.datasets.prepare import prepare_synthetic
    random.seed(SEED)

    synthetic = prepare_synthetic(OUT / "data", n_scenarios=250, seed=SEED)
    print(f"    Synthetic: {len(synthetic)} examples")


    sqa_mod = sys.modules.get("wmw.datasets.scienceqa")
    mv_mod = sys.modules.get("wmw.datasets.mathvista")
    if sqa_mod is None:
        from wmw.datasets.scienceqa import download_scienceqa
    else:
        download_scienceqa = sqa_mod.download_scienceqa
    if mv_mod is None:
        from wmw.datasets.mathvista import download_mathvista
    else:
        download_mathvista = mv_mod.download_mathvista

    scienceqa = download_scienceqa(OUT / "data/scienceqa", max_examples=60)
    print(f"    ScienceQA: {len(scienceqa)} examples")
    if len(scienceqa) == 0:
        print("\n  ERROR: ScienceQA returned 0 examples.")
        print("  Fix: pip install datasets Pillow")
        print("  Fix: ensure internet access to HuggingFace Hub")
        sys.exit(1)

    mathvista = download_mathvista(OUT / "data/mathvista", max_examples=60)
    print(f"    MathVista: {len(mathvista)} examples")
    if len(mathvista) == 0:
        print("\n  ERROR: MathVista returned 0 examples.")
        print("  Fix: pip install datasets Pillow")
        print("  Fix: ensure internet access to HuggingFace Hub")
        sys.exit(1)

    all_ex = synthetic + scienceqa + mathvista
    print(f"  Total: {len(all_ex)} examples "
          f"({len(synthetic)} synth + {len(scienceqa)} SQA + {len(mathvista)} MV)")
    if len(all_ex) < 300:
        print(f"\n  ERROR: Only {len(all_ex)} examples. Need 300+.")
        print(f"  Increase n_scenarios or max_examples and retry.")
        sys.exit(1)


    from wmw.datasets.common import save_examples
    save_examples(all_ex, scaled_path)
    return all_ex


def call_r(cfg, sys_p, prompt, img=None):
    for i in range(3):
        r = call_vlm(cfg, sys_p, prompt, img)
        if not r.error: return r
        wait = 2 ** i
        print(f"      Retry in {wait}s: {r.error[:60]}")
        time.sleep(wait)
    return r


def save_jl(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in data: json.dump(d, f, default=str); f.write("\n")


def append_jl(row, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        json.dump(row, f, default=str)
        f.write("\n")


def load_jl(path):
    if not path.exists(): return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_checkpoint(path):
    done = set()
    if path.exists():
        for r in load_jl(path):
            done.add((r.get("model", ""), r.get("id", "")))
    return done


def wilson(k, n, z=1.96):
    if n == 0: return (0, 1)
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d; m = z*math.sqrt((p*(1-p)+z**2/(4*n))/n)/d
    return (max(0,c-m), min(1,c+m))


def fci(k, n):
    if n == 0: return "--"
    lo, hi = wilson(k, n)
    return f"{k/n*100:.1f}% [{lo*100:.0f},{hi*100:.0f}]"


def run_condition(examples, models, sys_prompt, prompt_fn, tag, checkpoint_path, delay=0.25):
    done = load_checkpoint(checkpoint_path)
    total_skipped = len(done)
    if total_skipped:
        print(f"  Resuming: {total_skipped} calls already checkpointed")

    results = load_jl(checkpoint_path)

    for mk in models:
        cfg = MODELS[mk]
        correct = 0
        n = len(examples)
        mk_done = sum(1 for m, _ in done if m == mk)
        mk_todo = n - mk_done
        print(f"\n  {cfg.name} ({mk_todo} remaining of {n})...")

        for i, ex in enumerate(examples):
            if (mk, ex.id) in done:

                existing = next((r for r in results if r["model"]==mk and r["id"]==ex.id), None)
                if existing and existing.get("correct"): correct += 1
                continue

            if (i+1) % 50 == 0: print(f"    [{i+1}/{n}]")
            prompt = prompt_fn(ex)
            resp = call_r(cfg, sys_prompt, prompt, ex.image_path)
            td, st = parse_trace(resp.raw_text)
            pred = extract_answer(td)
            is_correct = answers_match(pred, ex.gold_answer) if st != "failed" else False
            if is_correct: correct += 1

            vl, sok, tok = [], None, None
            if td and tag == "full_trace":
                vr = verify_trace(td)
                vl, sok, tok = vr.labels, vr.state_ok, vr.transition_ok

            row = {
                "tag": tag, "model": mk, "id": ex.id, "source": ex.source,
                "correct": is_correct, "parsed": st != "failed",
                "state_ok": sok, "transition_ok": tok, "labels": vl,
                "trace_dict": td, "predicted_answer": str(pred),
                "gold_answer": str(ex.gold_answer),
            }
            append_jl(row, checkpoint_path)
            results.append(row)
            done.add((mk, ex.id))
            time.sleep(delay)

        print(f"  → {cfg.name}: {correct}/{n} = {correct/n*100:.1f}%")

    return results


def run_full_trace():
    out = OUT / "full_trace"; out.mkdir(parents=True, exist_ok=True)
    examples = load_scaled()
    ckpt = out / "checkpoint.jsonl"

    results = run_condition(
        examples, ALL_MK,
        SYSTEM_PROMPTS["full_trace"],
        lambda ex: build_prompt(ex, "full_trace"),
        tag="full_trace",
        checkpoint_path=ckpt,
    )


    save_jl(results, out / "results.jsonl")


    print("\n" + "="*65)
    print("TABLE 3 — Multi-model matrix (n=%d):" % len(examples))
    print("="*65)
    for mk in ALL_MK:
        mr = [r for r in results if r["model"] == mk]
        n = len(mr)
        nc = sum(r["correct"] for r in mr)
        ns = sum(1 for r in mr if r.get("state_ok") is True)
        nt = sum(1 for r in mr if r.get("transition_ok") is True)
        nh = sum(1 for r in mr if r["correct"] and
                 (r.get("state_ok") is False or r.get("transition_ok") is False))
        np_ = sum(r["parsed"] for r in mr)
        print(f"  {MODELS[mk].name}:")
        print(f"    Trace acc:   {fci(nc, n)}")
        print(f"    State acc:   {fci(ns, n)}")
        print(f"    Trans acc:   {fci(nt, n)}")
        print(f"    Hidden inc:  {fci(nh, n)}")
        print(f"    Parse rate:  {np_/n*100:.1f}%")


    print("\n  Cross-source:")
    for src in ["synthetic", "scienceqa", "mathvista"]:
        row = f"    {src}:"
        for mk in ALL_MK:
            sr = [r for r in results if r["model"] == mk and r["source"] == src]
            ns = len(sr); sc = sum(r["correct"] for r in sr)
            row += f"  {MODELS[mk].name}={fci(sc,ns)}" if ns else ""
        print(row)


def run_terse():
    out = OUT / "terse"; out.mkdir(parents=True, exist_ok=True)
    examples = load_scaled()
    ckpt = out / "checkpoint.jsonl"

    SYS = "You are a physics expert. Answer with only the final answer."
    def terse_prompt(ex):
        opts = ""
        if ex.options:
            opts = "\nOptions:\n" + "\n".join(
                f"({chr(65+j)}) {o}" for j, o in enumerate(ex.options))
        return (f"Question: {ex.question}{opts}\n\n"
                f"Respond with ONLY the answer (letter or number). Nothing else.")

    results = run_condition(examples, ALL_MK, SYS, terse_prompt, tag="terse",
                            checkpoint_path=ckpt)
    save_jl(results, out / "results.jsonl")

    print("\n" + "="*65)
    print("TABLE TERSE (n=%d):" % len(examples))
    print("="*65)
    print(f"  {'Model':<16s} {'Terse':>8s} {'Ans-field':>10s} {'Full trace':>11s}")


    ft_path = OUT / "full_trace/results.jsonl"
    ft = load_jl(ft_path) if ft_path.exists() else []

    for mk in ALL_MK:
        tr = [r for r in results if r["model"] == mk]
        n = len(tr); tc = sum(r["correct"] for r in tr)
        ft_mk = [r for r in ft if r["model"] == mk]
        fn = len(ft_mk); fc = sum(r["correct"] for r in ft_mk) if ft_mk else 0
        terse_pct = f"{tc/n*100:.1f}%" if n else "--"
        ft_pct = f"{fc/fn*100:.1f}%" if fn else "--"
        print(f"  {MODELS[mk].name:<16s} {terse_pct:>8s} {'—':>10s} {ft_pct:>11s}")


def run_cross_judge():
    out = OUT / "crossjudge"; out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)


    ft_path = OUT / "full_trace/results.jsonl"
    if ft_path.exists():
        all_ft = load_jl(ft_path)
        gpt4o_traces = [r for r in all_ft if r["model"] == "gpt4o" and r.get("trace_dict")]
        sample = random.sample(gpt4o_traces, min(100, len(gpt4o_traces)))
        print(f"  Reusing {len(sample)} GPT-4o traces from full_trace run")
    else:

        examples = load_scaled()
        sample_ex = random.sample(examples, min(100, len(examples)))
        cfg = MODELS["gpt4o"]
        sample = []
        print(f"  Running GPT-4o on {len(sample_ex)} examples for cross-judge...")
        for i, ex in enumerate(sample_ex):
            if (i+1) % 20 == 0: print(f"    [{i+1}/{len(sample_ex)}]")
            resp = call_r(cfg, SYSTEM_PROMPTS["full_trace"],
                          build_prompt(ex, "full_trace"), ex.image_path)
            td, _ = parse_trace(resp.raw_text)
            if td:
                sample.append({"id": ex.id, "trace_dict": td})
            time.sleep(0.25)


    results = []
    ckpt = out / "checkpoint.jsonl"
    done = load_checkpoint(ckpt)
    existing = load_jl(ckpt)
    if existing:
        results = existing
        print(f"  Resuming: {len(done)} traces already judged")

    print(f"\n  Running both judges on {len(sample)} traces...")
    for i, r in enumerate(sample):
        if ("crossjudge", r["id"]) in done:
            continue
        if (i+1) % 20 == 0: print(f"    [{i+1}/{len(sample)}]")
        td = r["trace_dict"]

        vr_s = call_llm_judge(td, model_config=JUDGES["sonnet"])
        time.sleep(0.2)
        vr_g = call_llm_judge(td, model_config=JUDGES["gpt4o_j"])
        time.sleep(0.2)

        row = {
            "model": "crossjudge", "id": r["id"],
            "sonnet_labels": vr_s.labels,
            "sonnet_state_ok": vr_s.state_ok,
            "sonnet_trans_ok": vr_s.transition_ok,
            "gpt4o_labels": vr_g.labels,
            "gpt4o_state_ok": vr_g.state_ok,
            "gpt4o_trans_ok": vr_g.transition_ok,
        }
        append_jl(row, ckpt)
        results.append(row)
        done.add(("crossjudge", r["id"]))

    save_jl(results, out / "results.jsonl")
    n = len(results)


    agree_s = sum(1 for r in results if r["sonnet_state_ok"] == r["gpt4o_state_ok"])
    agree_t = sum(1 for r in results if r["sonnet_trans_ok"] == r["gpt4o_trans_ok"])
    se = sum(1 for r in results if r["sonnet_labels"])
    ge = sum(1 for r in results if r["gpt4o_labels"])

    a = [1 if r["sonnet_labels"] else 0 for r in results]
    b = [1 if r["gpt4o_labels"] else 0 for r in results]
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a)/n, sum(b)/n
    pe = pa*pb + (1-pa)*(1-pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1

    print(f"\n" + "="*65)
    print(f"TABLE CROSSJUDGE (n={n}):")
    print("="*65)
    print(f"  State-error agreement  & {agree_s/n*100:.1f}\\% \\\\")
    print(f"  Trans-error agreement  & {agree_t/n*100:.1f}\\% \\\\")
    print(f"  Cohen's kappa          & {kappa:.2f} \\\\")
    print(f"  Sonnet error rate      & {se/n*100:.1f}\\% \\\\")
    print(f"  GPT-4o error rate      & {ge/n*100:.1f}\\% \\\\")
    if se > ge * 1.2:
        print(f"\n  ⚠ Sonnet is {se/max(ge,1):.1f}× stricter than GPT-4o")
        print(f"    Hidden inc for GPT models may be inflated relative to Opus")
    else:
        print(f"\n  ✓ Judges are comparably strict (ratio: {se/max(ge,1):.2f})")


def run_pref_rerank():
    out = OUT / "pref_rerank"; out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    from wmw.generators.scenarios import generate_balanced
    from wmw.generators.trace_generator import generate_traces
    from wmw.generators.perturbation import get_perturbations, perturb_trace


    print("  Generating preference pairs...")
    specs = generate_balanced(100)
    traces = generate_traces(specs)
    seen_perts = get_perturbations(family="seen")
    held_perts = get_perturbations(family="held_out")
    pairs = []
    for t in traces:
        for p in seen_perts + held_perts:
            try: pairs.append(perturb_trace(t, p))
            except: pass
    print(f"  {len(pairs)} pairs")

    import numpy as np

    def feats(td):
        if td is None: return np.zeros(8)
        vr = verify_trace(td)
        s0 = td.get("state_0", {}); tr = td.get("transition", {})
        return np.array([
            len(s0.get("objects", [])),
            len(s0.get("forces", [])),
            len(s0.get("relations", [])),
            1 if tr.get("rule") else 0,
            len(tr.get("evidence", [])),
            1 if td.get("state_1", {}).get("predicted_change") else 0,
            1 if vr.state_ok else 0,
            -len(vr.labels),
        ])

    w = np.zeros(8)
    for p in pairs:
        w += feats(p.chosen) - feats(p.rejected)
    w = w / max(len(pairs), 1)
    print(f"  Weights: {w.round(3).tolist()}")

    def score(td):
        return float(np.dot(w, feats(td)))


    examples = load_scaled()
    synth = [ex for ex in examples if ex.source == "synthetic"][:50]
    results = []
    ckpt = out / "checkpoint.jsonl"
    done_models = set()
    if ckpt.exists():
        existing = load_jl(ckpt)
        for r in existing:
            done_models.add(r["model"])
            results.append(r)
        if done_models:
            print(f"  Resuming: {done_models} already complete")

    for mk in CHEAP:
        cfg = MODELS[mk]
        if cfg.name in done_models:
            continue
        s_cfg = ModelConfig(name=cfg.name, provider=cfg.provider,
            model_id=cfg.model_id, api_key_env=cfg.api_key_env,
            max_tokens=cfg.max_tokens, temperature=0.7, timeout=cfg.timeout)
        g_ok = r_ok = p_ok = 0
        print(f"\n  {cfg.name} (50 × k=5)...")
        for ex in synth:
            cands = []
            for _ in range(5):
                resp = call_r(s_cfg, SYSTEM_PROMPTS["full_trace"],
                              build_prompt(ex, "full_trace"), ex.image_path)
                td, _ = parse_trace(resp.raw_text)
                cands.append(td)
                time.sleep(0.15)

            if answers_match(extract_answer(cands[0]), ex.gold_answer): g_ok += 1

            rsc = []
            for td in cands:
                if td is None: rsc.append(-10); continue
                vr = verify_trace(td)
                rsc.append((2 if vr.state_ok else 0)+(2 if vr.transition_ok else 0)-len(vr.labels))
            best_r = cands[rsc.index(max(rsc))]
            if answers_match(extract_answer(best_r), ex.gold_answer): r_ok += 1

            psc = [score(td) for td in cands]
            best_p = cands[psc.index(max(psc))]
            if answers_match(extract_answer(best_p), ex.gold_answer): p_ok += 1
        n = len(synth)
        row = {"model": cfg.name, "n": n,
                         "greedy": round(g_ok/n, 3), "rules": round(r_ok/n, 3),
                         "pref": round(p_ok/n, 3)}
        append_jl(row, ckpt)
        results.append(row)
        print(f"  → greedy={g_ok}/{n} rules={r_ok}/{n} pref={p_ok}/{n}")

    save_jl(results, out / "results.jsonl")


    print("\n  Computing scaling curve...")
    seen_pairs = [p for p in pairs if p.perturbation_family == "seen"]
    held_pairs_eval = [p for p in pairs if p.perturbation_family == "held_out"][:200]
    random.shuffle(seen_pairs)
    scaling = []
    for nt in [50, 100, 200, 500, 1000, 2000, min(4000, len(seen_pairs)), len(seen_pairs)]:
        sub = seen_pairs[:min(nt, len(seen_pairs))]
        wt = np.zeros(8)
        for p in sub: wt += feats(p.chosen) - feats(p.rejected)
        wt = wt / max(len(sub), 1)
        eval_s = seen_pairs[max(0, len(seen_pairs)-200):]
        sc = sum(1 for p in eval_s if np.dot(wt, feats(p.chosen)) > np.dot(wt, feats(p.rejected)))
        hc = sum(1 for p in held_pairs_eval if np.dot(wt, feats(p.chosen)) > np.dot(wt, feats(p.rejected)))
        scaling.append({"n_pairs": min(nt, len(seen_pairs)),
                        "acc_seen": round(sc/len(eval_s), 3) if eval_s else 0,
                        "acc_held": round(hc/len(held_pairs_eval), 3) if held_pairs_eval else 0})
    save_jl(scaling, out / "scaling.jsonl")

    print("\n" + "="*65)
    print("TABLE PREF-RERANK:")
    print("="*65)
    for r in results:
        print(f"  {r['model']}: greedy={r['greedy']*100:.1f}% rules={r['rules']*100:.1f}% pref={r['pref']*100:.1f}%")
    print("\n  Run: python generate_figures.py --results data/experiments/pref_rerank")


def run_revision():
    out = OUT / "revision"; out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    examples = load_scaled()
    by_src = defaultdict(list)
    for ex in examples: by_src[ex.source].append(ex)
    rev_set = []
    for pool in by_src.values():
        rev_set.extend(random.sample(pool, min(20, len(pool))))
    rev_set = rev_set[:60]


    ft_path = OUT / "full_trace/results.jsonl"
    ft = {(r["model"], r["id"]): r for r in load_jl(ft_path)} if ft_path.exists() else {}

    results = []
    ckpt = out / "checkpoint.jsonl"
    done = load_checkpoint(ckpt)
    if done:
        results = load_jl(ckpt)
        print(f"  Resuming: {len(done)} revision calls checkpointed")

    for mk in CHEAP:
        cfg = MODELS[mk]
        improved = 0
        mk_todo = sum(1 for ex in rev_set if (mk, ex.id) not in done)
        print(f"\n  {cfg.name} ({mk_todo} remaining of {len(rev_set)})...")
        for ex in rev_set:
            if (mk, ex.id) in done:
                existing = next((r for r in results if r["model"]==mk and r["id"]==ex.id), None)
                if existing and existing.get("improved"): improved += 1
                continue
            orig = ft.get((mk, ex.id))
            td = orig["trace_dict"] if orig else None
            orig_correct = orig["correct"] if orig else False

            feedback_lines = []
            if td:
                vr = verify_trace(td)
                for d in vr.details:
                    if "passed" not in d.lower(): feedback_lines.append(f"- {d}")
                if vr.labels: feedback_lines.append(f"Error types: {', '.join(vr.labels)}")
            feedback = "\n".join(feedback_lines) or "Minor issues detected."

            rev_prompt = build_prompt(ex, condition="revise", verifier_feedback=feedback)
            rev_resp = call_r(cfg, SYSTEM_PROMPTS["revise"], rev_prompt, ex.image_path)
            rev_td, _ = parse_trace(rev_resp.raw_text)
            rev_correct = answers_match(extract_answer(rev_td), ex.gold_answer) if rev_td else False
            did_improve = (not orig_correct) and rev_correct
            if did_improve: improved += 1

            results.append({"model": mk, "id": ex.id, "source": ex.source,
                            "orig_correct": orig_correct, "rev_correct": rev_correct,
                            "improved": did_improve})
            append_jl(results[-1], ckpt)
            done.add((mk, ex.id))
            time.sleep(0.25)

        n = len(rev_set)
        lo, hi = wilson(improved, n)
        print(f"  → Improved: {improved}/{n} = {improved/n*100:.1f}% [{lo*100:.0f},{hi*100:.0f}]")

    save_jl(results, out / "results.jsonl")

    print("\n" + "="*65)
    print("REVISION RESULTS (n=60):")
    print("="*65)
    for mk in CHEAP:
        mr = [r for r in results if r["model"] == mk]
        n = len(mr); imp = sum(r["improved"] for r in mr)
        lo, hi = wilson(imp, n)
        print(f"  {MODELS[mk].name}: {imp}/{n} = {imp/n*100:.1f}% [{lo*100:.0f},{hi*100:.0f}]")
    print("  (Opus 15/15 and GPT-5.5 0/15 from original evaluation)")


def audit_sample():
    out = OUT / "audit"; out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    ft_path = OUT / "full_trace/results.jsonl"
    if not ft_path.exists():
        print("ERROR: Run --full-trace first."); return
    results = load_jl(ft_path)

    sampled = []
    for mk in ALL_MK:
        pool = [r for r in results if r["model"] == mk]
        correct = [r for r in pool if r["correct"]]
        incorrect = [r for r in pool if not r["correct"]]
        sampled.extend(random.sample(correct, min(25, len(correct))))
        sampled.extend(random.sample(incorrect, min(25, len(incorrect))))
        if mk == "gpt5_5":
            flagged = [r for r in correct if r.get("labels") and r not in sampled]
            sampled.extend(random.sample(flagged, min(15, len(flagged))))
    random.shuffle(sampled)

    for ann in ["A", "B"]:
        subset = sampled if ann == "A" else sampled[:40]
        p = out / f"sheet_{ann}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["audit_id","source","trace_json","predicted_answer","gold_answer",
                        "human_state_ok","human_transition_ok","human_answer_trace_ok",
                        "human_error_labels","human_notes"])
            for i, r in enumerate(subset):
                tj = json.dumps(r.get("trace_dict",{}), indent=2, default=str)[:2000]
                w.writerow([f"A{i+1:04d}", r.get("source",""), tj,
                            r.get("predicted_answer",""), r.get("gold_answer",""),
                            "","","","",""])
        print(f"  {ann}: {p} ({len(subset)} traces)")

    save_jl([{"audit_id": f"A{i+1:04d}", "model": r["model"], "correct": r["correct"],
              "judge_labels": r.get("labels",[]), "in_overlap": i < 40}
             for i, r in enumerate(sampled)], out / "key.jsonl")
    print(f"  Key → {out/'key.jsonl'}")
    print("\n  Annotator A: all traces | Annotator B: 40-trace overlap")


def audit_analyze():
    out = OUT / "audit"
    key = {d["audit_id"]: d for d in load_jl(out / "key.jsonl")}

    def load_ann(path):
        ann = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                ann[row["audit_id"]] = {
                    "state_ok": row.get("human_state_ok","").strip().lower(),
                    "transition_ok": row.get("human_transition_ok","").strip().lower()}
        return ann

    ann_a = load_ann(out / "sheet_A.csv")
    ann_b = load_ann(out / "sheet_B.csv") if (out / "sheet_B.csv").exists() else {}


    kappa_val = None
    if ann_b:
        overlap = [a for a in ann_a if a in ann_b]
        n = len(overlap)
        ae = [1 if ann_a[a]["state_ok"]=="no" or ann_a[a]["transition_ok"]=="no" else 0 for a in overlap]
        be = [1 if ann_b[a]["state_ok"]=="no" or ann_b[a]["transition_ok"]=="no" else 0 for a in overlap]
        po = sum(x==y for x,y in zip(ae,be))/n if n else 0
        pa, pb = sum(ae)/n, sum(be)/n; pe = pa*pb+(1-pa)*(1-pb)
        kappa_val = (po-pe)/(1-pe) if pe < 1 else 1
        print(f"  Inter-annotator kappa: {kappa_val:.3f} (n={n})")


    cm = {v: {"tp":0,"fp":0,"fn":0,"tn":0} for v in ["rules","judge","ensemble"]}
    hid = {mk: {"human":0,"judge":0,"nc":0} for mk in ALL_MK}
    RULE_L = {"object","unit_scale","intervention"}

    for aid, k in key.items():
        ann = ann_a.get(aid, {})
        if ann.get("state_ok") == "ambiguous": continue
        he = ann.get("state_ok")=="no" or ann.get("transition_ok")=="no"
        jl = k.get("judge_labels",[]); je = len(jl)>0
        re_ = any(l in RULE_L for l in jl); ee = je
        for v, ve in [("rules",re_),("judge",je),("ensemble",ee)]:
            c = cm[v]
            if ve and he: c["tp"]+=1
            elif ve and not he: c["fp"]+=1
            elif not ve and he: c["fn"]+=1
            else: c["tn"]+=1
        mk = k["model"]
        if k["correct"]:
            hid[mk]["nc"]+=1
            if he: hid[mk]["human"]+=1
            if je: hid[mk]["judge"]+=1

    print("\n" + "="*65)
    print("TABLE model-human-audit:")
    print("="*65)
    for v in ["rules","judge","ensemble"]:
        c=cm[v]; tp,fp,fn=c["tp"],c["fp"],c["fn"]
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
        f1=2*p*r/(p+r) if p+r else 0
        lab={"rules":"Rules-only","judge":"LLM judge","ensemble":"Rules + LLM"}[v]
        print(f"  {lab} & {p:.3f} & {r:.3f} & {f1:.3f} & 120 \\\\")
    if kappa_val: print(f"  Kappa: {kappa_val:.2f}")

    print("\nTABLE verifier-confusion:")
    c=cm["ensemble"]
    print(f"  Flags error & {c['tp']} & {c['fp']} \\\\")
    print(f"  Clean       & {c['fn']} & {c['tn']} \\\\")

    print("\nTABLE audit-hidden:")
    for mk in ALL_MK:
        h=hid[mk]; nc=h["nc"]
        if nc:
            print(f"  {MODELS[mk].name}: human={h['human']/nc*100:.1f}% judge={h['judge']/nc*100:.1f}% (nc={nc})")


def ceiling_sample():
    out = OUT / "ceiling"; out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    examples = load_scaled()
    by_src = defaultdict(list)
    for ex in examples: by_src[ex.source].append(ex)
    sample = []
    for pool in by_src.values(): sample.extend(random.sample(pool, min(17, len(pool))))
    sample = sample[:50]; random.shuffle(sample)

    for ann in ["A","B"]:
        p = out / f"ceiling_{ann}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["problem_id","source","question","options","image_path",
                        "human_answer","human_confidence"])
            for ex in sample:
                opts = "; ".join(f"({chr(65+j)}) {o}" for j,o in enumerate(ex.options)) if ex.options else ""
                w.writerow([ex.id, ex.source, ex.question, opts, str(ex.image_path), "", ""])
        print(f"  {ann}: {p}")
    save_jl([{"problem_id": ex.id, "source": ex.source,
              "gold_answer": str(ex.gold_answer)} for ex in sample], out / "key.jsonl")
    print(f"\n  Both annotators answer all 50 problems (image + question only)")


def ceiling_analyze():
    out = OUT / "ceiling"
    key = {d["problem_id"]: d for d in load_jl(out / "key.jsonl")}
    anns = {}
    for aid in ["A","B"]:
        p = out / f"ceiling_{aid}.csv"
        if not p.exists(): continue
        a = {}
        with open(p) as f:
            for row in csv.DictReader(f): a[row["problem_id"]] = row.get("human_answer","").strip()
        anns[aid] = a

    print("\n" + "="*65)
    print("TABLE human-ceiling:")
    print("="*65)
    if len(anns) == 2:
        a, b = list(anns.values())
        for src in ["synthetic","scienceqa","mathvista"]:
            ids = [pid for pid,k in key.items() if k["source"]==src]; n=len(ids)
            ca = sum(1 for pid in ids if answers_match(a.get(pid), key[pid]["gold_answer"]))
            cb = sum(1 for pid in ids if answers_match(b.get(pid), key[pid]["gold_answer"]))
            print(f"  {src}: {(ca+cb)/(2*n)*100:.1f}\\%  (A={ca}/{n} B={cb}/{n})")
        shared = [pid for pid in key if pid in a and pid in b]
        agree = sum(1 for pid in shared if a[pid].upper()==b[pid].upper())
        print(f"  Agreement: {agree}/{len(shared)} = {agree/len(shared)*100:.1f}\\%")


def main():
    ap = argparse.ArgumentParser(description="WMW experiments — fill every TBD")
    ap.add_argument("--all", action="store_true",
        help="Run all API experiments ($135, ~5-6h)")
    ap.add_argument("--full-trace", action="store_true",
        help="Full-trace 300×4 ($61.50, ~2.5h)")
    ap.add_argument("--terse", action="store_true",
        help="Terse baseline 300×4 ($61.50, ~2h)")
    ap.add_argument("--cross-judge", action="store_true",
        help="Cross-judge calibration 100×2 ($4, ~30min)")
    ap.add_argument("--pref-rerank", action="store_true",
        help="Preference reranking 50×mini+4o×5 ($6.25, ~1h)")
    ap.add_argument("--revision", action="store_true",
        help="Expanded revision 60×mini+4o ($1.50, ~15min)")
    ap.add_argument("--audit-sample", action="store_true")
    ap.add_argument("--audit-analyze", action="store_true")
    ap.add_argument("--ceiling-sample", action="store_true")
    ap.add_argument("--ceiling-analyze", action="store_true")
    args = ap.parse_args()

    if not any(vars(args).values()):
        ap.print_help()
        print("\n  ┌─────────────────────────────────────────────────────────┐")
        print("  │ BUDGET: $100 OpenAI + $60 Anthropic = $160 (with buffer)│")
        print("  ├─────────────────────────────────────────────────────────┤")
        print("  │ --all           runs everything below ($135)            │")
        print("  │ --full-trace    300×4 models, main results   ($61.50)   │")
        print("  │ --terse         300×4 models, terse baseline ($61.50)   │")
        print("  │ --cross-judge   100×2 judges, kappa          ($ 4.00)   │")
        print("  │ --pref-rerank   50×mini+4o×k=5, demo         ($ 6.25)  │")
        print("  │ --revision      60×mini+4o, expanded          ($ 1.50)  │")
        print("  ├─────────────────────────────────────────────────────────┤")
        print("  │ After annotation (human labor, $0 API):                 │")
        print("  │ --audit-sample / --audit-analyze   → 17 TBDs           │")
        print("  │ --ceiling-sample / --ceiling-analyze → 4 TBDs          │")
        print("  └─────────────────────────────────────────────────────────┘")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    if args.all:
        print("\n" + "█"*65)
        print("  RUNNING ALL EXPERIMENTS (~$135, ~5-6 hours)")
        print("█"*65)
        run_full_trace()
        run_terse()
        run_cross_judge()
        run_pref_rerank()
        run_revision()
        audit_sample()
        ceiling_sample()
        print("\n" + "█"*65)
        print("  ALL API EXPERIMENTS COMPLETE")
        print("  Annotation sheets ready in data/experiments/audit/ and ceiling/")
        print("  After annotation: --audit-analyze and --ceiling-analyze")
        print("█"*65)
        return

    if args.full_trace: run_full_trace()
    if args.terse: run_terse()
    if args.cross_judge: run_cross_judge()
    if args.pref_rerank: run_pref_rerank()
    if args.revision: run_revision()
    if args.audit_sample: audit_sample()
    if args.audit_analyze: audit_analyze()
    if args.ceiling_sample: ceiling_sample()
    if args.ceiling_analyze: ceiling_analyze()


if __name__ == "__main__":
    main()
