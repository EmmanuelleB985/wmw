from __future__ import annotations
import re
from wmw.schemas.models import VerifierResult


_POSITIVE_DIRECTIONS = {"right", "rightward", "upward", "up", "forward", "ascending", "increases", "accelerates"}
_NEGATIVE_DIRECTIONS = {"left", "leftward", "downward", "down", "backward", "descending", "decreases", "decelerates"}


_EXPECTED_RULES: dict[str, set[str]] = {
    "inclined_plane": {"newton", "second law", "incline", "force", "component"},
    "projectile": {"kinematic", "projectile", "gravity", "constant acceleration"},
    "collision": {"momentum", "conservation", "kinetic energy"},
    "pulley": {"atwood", "newton", "tension", "pulley"},
    "spring": {"hooke", "spring", "elastic", "potential"},
    "circuit": {"ohm", "kirchhoff", "resistance", "series", "parallel"},
    "free_fall": {"free fall", "kinematic", "gravity", "h = ", "gt"},
    "friction": {"friction", "static", "kinetic", "normal", "applied"},
    "circular_motion": {"centripetal", "circular", "v²/r"},
    "pendulum": {"pendulum", "period", "simple harmonic", "√(l/g)"},
    "lever": {"torque", "lever", "moment", "balance", "fulcrum"},
    "buoyancy": {"archimedes", "buoyancy", "density", "displacement"},
    "thermal": {"q = mc", "specific heat", "thermal", "calorimetry"},
    "wave": {"wave", "v = fλ", "frequency", "wavelength"},
    "optics": {"lens", "refraction", "1/f", "thin lens", "snell"},
    "fluid": {"hydrostatic", "pressure", "ρgh", "pascal"},
    "em_induction": {"faraday", "induction", "emf", "flux", "dΦ/dt"},
}


def _extract_sign(text: str) -> int | None:
    lower = text.lower()
    if any(d in lower for d in _POSITIVE_DIRECTIONS):
        return 1
    if any(d in lower for d in _NEGATIVE_DIRECTIONS):
        return -1
    return None


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in re.findall(r'-?\d+\.?\d*', text)]


def verify_transition(trace_dict: dict) -> VerifierResult:
    details: list[str] = []
    labels: list[str] = []
    abstained = False

    s0 = trace_dict.get("state_0", {})
    tr = trace_dict.get("transition", {})
    s1 = trace_dict.get("state_1", {})
    ans = trace_dict.get("answer", {})
    family = trace_dict.get("scenario_family", "")


    rule = tr.get("rule", "").lower()
    expected = _EXPECTED_RULES.get(family, set())
    if expected and not any(kw in rule for kw in expected):
        details.append(
            f"rule '{tr.get('rule', '')}' may not match family '{family}'; "
            f"expected keywords: {expected}"
        )
        if "transition" not in labels:
            labels.append("transition")


    forces = s0.get("forces", [])
    effect = tr.get("effect", "")
    effect_sign = _extract_sign(effect)

    if forces and effect_sign is not None:

        net_dir = None
        for f in forces:
            if not isinstance(f, dict):
                continue
            fname = f.get("name", "").lower()
            if fname in ("applied", "net", "resultant"):
                net_dir = _extract_sign(f.get("direction", ""))
                break
        if net_dir is not None and net_dir != effect_sign:
            details.append(
                f"net force direction (sign={net_dir}) disagrees with "
                f"transition effect direction (sign={effect_sign})"
            )
            if "transition" not in labels:
                labels.append("transition")


    predicted_change = s1.get("predicted_change", "")
    transition_sign = _extract_sign(effect)
    result_sign = _extract_sign(predicted_change)

    if transition_sign is not None and result_sign is not None:
        if transition_sign != result_sign:
            details.append(
                f"transition effect sign ({transition_sign}) disagrees with "
                f"predicted_change sign ({result_sign})"
            )
            if "intervention" not in labels:
                labels.append("intervention")


    evidence = tr.get("evidence", [])
    all_text = " ".join([effect] + evidence + [predicted_change]).lower()
    temporal_before = ["before", "initial", "pre-collision", "at t=0"]
    temporal_after = ["after", "final", "post-collision", "resulting"]


    s0_text = str(s0.get("variables", {})).lower()
    s1_text = predicted_change.lower()
    if any(t in s0_text for t in ["[swapped]", "was result", "post-collision", "final state"]):
        details.append("state_0 appears to contain post-transition information")
        if "temporal" not in labels:
            labels.append("temporal")
    if any(t in s1_text for t in ["[swapped]", "was initial", "pre-collision"]):
        details.append("state_1 appears to contain pre-transition information")
        if "temporal" not in labels:
            labels.append("temporal")


    equation = tr.get("equation", "")
    if equation:

        if equation.count("(") != equation.count(")"):
            details.append(f"unbalanced parentheses in equation: '{equation}'")
            if "transition" not in labels:
                labels.append("transition")

        if "[REVERSED]" in equation or "[INVERTED]" in equation:
            details.append(f"equation contains perturbation marker")
            if "transition" not in labels:
                labels.append("transition")


    answer_trace_ok = True
    answer_val = ans.get("value")
    answer_explanation = ans.get("explanation", "")


    if "[FAITHFULNESS ERROR]" in str(answer_explanation):
        details.append("answer contains faithfulness error marker")
        labels.append("faithfulness")
        answer_trace_ok = False


    if isinstance(answer_val, str):
        ans_sign = _extract_sign(answer_val)
        if ans_sign is not None and transition_sign is not None and ans_sign != transition_sign:

            if family in ("buoyancy", "friction"):

                change_lower = predicted_change.lower()
                ans_lower = answer_val.lower()
                if ("floats" in change_lower and "sinks" in ans_lower) or\
                   ("sinks" in change_lower and "floats" in ans_lower):
                    details.append(f"answer '{answer_val}' contradicts predicted change '{predicted_change}'")
                    labels.append("faithfulness")
                    answer_trace_ok = False

    if isinstance(answer_val, (int, float)):

        new_vars = s1.get("new_variables", {})
        numeric_results = [v for v in new_vars.values() if isinstance(v, (int, float))]
        if numeric_results and answer_val != 0:

            closest = min(numeric_results, key=lambda x: abs(x - answer_val) if isinstance(x, (int, float)) else float('inf'))
            if isinstance(closest, (int, float)) and closest != 0:
                ratio = abs(answer_val / closest) if closest != 0 else float('inf')
                if ratio > 100 or ratio < 0.01:
                    details.append(
                        f"answer value {answer_val} is far from nearest result variable {closest}"
                    )
                    if "faithfulness" not in labels:
                        labels.append("faithfulness")
                    answer_trace_ok = False


    if not details:
        details.append("transition: all checks passed")

    if not labels and not details[0].endswith("passed"):

        pass

    transition_ok = "transition" not in labels and "intervention" not in labels and "temporal" not in labels

    return VerifierResult(
        transition_ok=transition_ok,
        answer_trace_ok=answer_trace_ok,
        details=details,
        labels=labels,
        abstained=abstained,
    )
