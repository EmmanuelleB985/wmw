from __future__ import annotations
import copy
import json
import random
import uuid
from dataclasses import dataclass
from typing import Any

from wmw.schemas.models import (
    Trace, PreferencePair, Force, Relation, PhysicalObject, FAILURE_LABELS,
)


@dataclass
class Perturbation:
    label: str
    field: str
    family: str
    name: str
    apply: callable


_PERTURBATIONS: list[Perturbation] = []


def _register(label, field, family, name):
    def decorator(fn):
        _PERTURBATIONS.append(Perturbation(label, field, family, name, fn))
        return fn
    return decorator


_DIRECTION_OPPOSITES = {
    "downward": "upward", "upward": "downward",
    "leftward": "rightward", "rightward": "leftward",
    "left": "right", "right": "left",
    "horizontal (right)": "horizontal (left)",
    "horizontal (left)": "horizontal (right)",
    "horizontal (left, static)": "horizontal (right, static)",
    "toward center": "away from center",
    "away from center": "toward center",
    "toward equilibrium": "away from equilibrium",
    "along string toward pivot": "away from pivot",
    "down the incline": "up the incline",
    "up the incline": "down the incline",
    "perpendicular to incline surface": "parallel to incline surface",
}

_RELATION_SWAPS = {
    "on": "above",
    "above": "below",
    "below": "above",
    "in": "adjacent",
    "adjacent": "in",
    "left_of": "right_of",
    "right_of": "left_of",
    "contact": "separated",
    "separated": "contact",
    "supported_by": "above",
    "connected": "separated",
    "attached": "separated",
}

_UNIT_CONFUSIONS = {
    "m": "cm", "cm": "m",
    "m/s": "km/h", "km/h": "m/s",
    "m/s²": "cm/s²",
    "N": "kN", "kN": "N",
    "J": "kJ", "kJ": "J",
    "Pa": "kPa", "kPa": "Pa",
    "kg": "g", "g": "kg",
    "s": "ms", "ms": "s",
    "°C": "°F", "°F": "°C",
    "Hz": "kHz", "kHz": "Hz",
    "A": "mA", "mA": "A",
    "V": "mV", "mV": "V",
    "Ω": "kΩ", "kΩ": "Ω",
}


def _reverse_direction(d: str) -> str:
    dl = d.lower().strip()
    for k, v in _DIRECTION_OPPOSITES.items():
        if k.lower() in dl:
            return d.replace(k, v, 1).replace(k.capitalize(), v.capitalize(), 1)

    return f"reversed: {d}"


def _deep_copy(trace_dict: dict) -> dict:
    return json.loads(json.dumps(trace_dict))


@_register("force", "state_0", "seen", "reverse_force_direction")
def reverse_force_direction(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    forces = td["state_0"]["forces"]
    if not forces:
        return td, "no forces to reverse"
    f = random.choice(forces)
    old = f["direction"]
    f["direction"] = _reverse_direction(old)
    return td, f"Reversed force '{f['name']}' direction: '{old}' → '{f['direction']}'"


@_register("force", "state_0", "seen", "remove_force")
def remove_force(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    forces = td["state_0"]["forces"]
    if not forces:
        return td, "no forces to remove"
    f = random.choice(forces)
    forces.remove(f)
    return td, f"Removed force '{f['name']}' on '{f['target']}'"


@_register("force", "state_0", "held_out", "swap_force_target")
def swap_force_target(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    forces = td["state_0"]["forces"]
    objects = [o["name"] for o in td["state_0"]["objects"]]
    if not forces or len(objects) < 2:
        return td, "not enough objects to swap force target"
    f = random.choice(forces)
    old_target = f["target"]
    candidates = [o for o in objects if o != old_target]
    if not candidates:
        return td, "no alternative target"
    f["target"] = random.choice(candidates)
    return td, f"Moved force '{f['name']}' from '{old_target}' to '{f['target']}'"


@_register("force", "state_0", "held_out", "double_force_magnitude")
def double_force_magnitude(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    forces = td["state_0"]["forces"]
    numeric = [f for f in forces if isinstance(f.get("magnitude"), (int, float))]
    if not numeric:
        return td, "no numeric forces"
    f = random.choice(numeric)
    old = f["magnitude"]
    f["magnitude"] = round(old * 2, 2)
    return td, f"Doubled force '{f['name']}' magnitude: {old} → {f['magnitude']}"


@_register("relation", "state_0", "seen", "swap_relation_type")
def swap_relation_type(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    rels = td["state_0"]["relations"]
    if not rels:
        return td, "no relations to swap"
    r = random.choice(rels)
    old = r["type"]
    r["type"] = _RELATION_SWAPS.get(old, "separated")
    return td, f"Changed relation '{old}' → '{r['type']}' between {r['args']}"


@_register("relation", "state_0", "seen", "reverse_relation_args")
def reverse_relation_args(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    rels = td["state_0"]["relations"]
    if not rels:
        return td, "no relations to reverse"
    r = random.choice(rels)
    old_args = list(r["args"])
    r["args"] = [old_args[1], old_args[0]]
    return td, f"Reversed relation args: {old_args} → {r['args']}"


@_register("relation", "state_0", "held_out", "add_phantom_relation")
def add_phantom_relation(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    objects = [o["name"] for o in td["state_0"]["objects"]]
    if len(objects) < 2:
        return td, "not enough objects"
    a, b = random.sample(objects, 2)
    phantom = random.choice(["contact", "attached", "supported_by", "aligned"])
    td["state_0"]["relations"].append({"type": phantom, "args": [a, b]})
    return td, f"Added phantom relation '{phantom}' between {a} and {b}"


@_register("object", "state_0", "seen", "rename_object")
def rename_object(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    objects = td["state_0"]["objects"]
    if not objects:
        return td, "no objects"
    obj = random.choice(objects)
    old_name = obj["name"]
    confusables = {
        "block": "cart", "cart": "block", "ball": "sphere", "sphere": "ball",
        "object": "particle", "particle": "object", "bob": "weight",
        "mass_1": "mass_2", "mass_2": "mass_1", "object_A": "object_B",
        "object_B": "object_A", "R1": "R2", "R2": "R1", "load": "effort",
        "effort": "load",
    }
    obj["name"] = confusables.get(old_name, f"wrong_{old_name}")
    return td, f"Renamed '{old_name}' → '{obj['name']}' (object confusion)"


@_register("object", "state_0", "held_out", "merge_objects")
def merge_objects(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    objects = td["state_0"]["objects"]
    if len(objects) < 2:
        return td, "not enough objects to merge"
    a = objects[0]
    b = objects[1]
    merged_name = f"{a['name']}+{b['name']}"
    merged_attrs = {**a.get("attributes", {}), **b.get("attributes", {})}
    objects.remove(b)
    a["name"] = merged_name
    a["attributes"] = merged_attrs
    return td, f"Merged '{a['name']}' and '{b['name']}' into '{merged_name}'"


@_register("state", "state_0", "seen", "flip_variable_sign")
def flip_variable_sign(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    vs = td["state_0"]["variables"]
    numeric = [(k, v) for k, v in vs.items() if isinstance(v, (int, float)) and v != 0]
    if not numeric:
        return td, "no numeric variables to flip"
    k, v = random.choice(numeric)
    vs[k] = -v
    return td, f"Flipped sign of '{k}': {v} → {-v}"


@_register("state", "state_0", "seen", "zero_out_variable")
def zero_out_variable(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    vs = td["state_0"]["variables"]
    numeric = [(k, v) for k, v in vs.items() if isinstance(v, (int, float)) and v != 0]
    if not numeric:
        return td, "no numeric variables"
    k, v = random.choice(numeric)
    vs[k] = 0
    return td, f"Zeroed variable '{k}': {v} → 0"


@_register("state", "state_0", "held_out", "swap_two_variables")
def swap_two_variables(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    vs = td["state_0"]["variables"]
    numeric = [(k, v) for k, v in vs.items() if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return td, "not enough variables to swap"
    (k1, v1), (k2, v2) = random.sample(numeric, 2)
    vs[k1], vs[k2] = v2, v1
    return td, f"Swapped '{k1}'={v1} ↔ '{k2}'={v2}"


@_register("transition", "transition", "seen", "reverse_transition_effect")
def reverse_transition_effect(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old_effect = td["transition"]["effect"]
    replacements = [
        ("accelerates", "decelerates"), ("decelerates", "accelerates"),
        ("increases", "decreases"), ("decreases", "increases"),
        ("rises", "falls"), ("falls", "rises"),
        ("floats", "sinks"), ("sinks", "floats"),
        ("rightward", "leftward"), ("leftward", "rightward"),
        ("upward", "downward"), ("downward", "upward"),
        ("ascending", "descending"), ("descending", "ascending"),
        ("down the incline", "up the incline"), ("up the incline", "down the incline"),
    ]
    new_effect = old_effect
    for a, b in replacements:
        if a in new_effect.lower():
            new_effect = new_effect.replace(a, b, 1)
            break
    if new_effect == old_effect:
        new_effect = f"[REVERSED] {old_effect}"
    td["transition"]["effect"] = new_effect
    return td, f"Reversed transition effect: '{old_effect}' → '{new_effect}'"


@_register("transition", "transition", "seen", "wrong_physical_rule")
def wrong_physical_rule(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old_rule = td["transition"]["rule"]
    wrong_rules = [
        "Conservation of angular momentum",
        "Bernoulli's principle",
        "Ideal gas law: PV = nRT",
        "Faraday's law of induction",
        "Stefan-Boltzmann radiation law",
        "Snell's law of refraction",
        "Heisenberg uncertainty principle",
        "Lenz's law",
    ]
    new_rule = random.choice([r for r in wrong_rules if r != old_rule])
    td["transition"]["rule"] = new_rule
    return td, f"Wrong rule: '{old_rule}' → '{new_rule}'"


@_register("transition", "transition", "held_out", "corrupt_equation")
def corrupt_equation(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    eq = td["transition"].get("equation", "")
    if not eq:
        return td, "no equation to corrupt"
    old_eq = eq
    corruptions = [
        ("·", "/"), ("/", "·"), ("+", "-"), ("-", "+"),
        ("*", "/"), ("/", "*"), ("sin", "cos"), ("cos", "sin"),
        ("²", "³"), ("×", "/"),
    ]
    for a, b in corruptions:
        if a in eq:
            eq = eq.replace(a, b, 1)
            break
    if eq == old_eq:

        import re
        m = re.search(r'=\s*(-?\d+\.?\d*)', eq)
        if m:
            val = float(m.group(1))
            eq = eq[:m.start(1)] + str(round(-val, 4)) + eq[m.end(1):]
    td["transition"]["equation"] = eq
    return td, f"Corrupted equation: '{old_eq}' → '{eq}'"


@_register("intervention", "state_1", "seen", "invert_predicted_change")
def invert_predicted_change(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old = td["state_1"]["predicted_change"]
    replacements = [
        ("accelerates", "decelerates"), ("decelerates", "accelerates"),
        ("increases", "decreases"), ("decreases", "increases"),
        ("rises", "sinks"), ("sinks", "rises"),
        ("floats", "sinks"), ("sinks", "floats"),
        ("moves", "stays stationary"),
    ]
    new = old
    for a, b in replacements:
        if a in new.lower():
            new = new.replace(a, b, 1)
            break
    if new == old:
        new = f"[INVERTED] {old}"
    td["state_1"]["predicted_change"] = new
    return td, f"Inverted predicted change: '{old}' → '{new}'"


@_register("intervention", "state_1", "held_out", "randomize_new_variables")
def randomize_new_variables(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    nvs = td["state_1"].get("new_variables", {})
    if not nvs:
        return td, "no new variables"
    changed = []
    for k, v in nvs.items():
        if isinstance(v, (int, float)):
            old = v
            nvs[k] = round(random.uniform(-abs(v) * 3, abs(v) * 3), 2)
            changed.append(f"{k}: {old} → {nvs[k]}")
    return td, f"Randomized resulting variables: {'; '.join(changed) or 'none changed'}"


@_register("temporal", "state_0", "seen", "swap_before_after")
def swap_before_after(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old_s0 = td["state_0"]
    old_s1 = td["state_1"]

    td["state_0"]["variables"]["note"] = f"[SWAPPED] was result: {old_s1['predicted_change']}"
    td["state_1"]["predicted_change"] = f"[SWAPPED] was initial state"
    return td, "Swapped before/after temporal framing"


@_register("temporal", "transition", "held_out", "reverse_temporal_order")
def reverse_temporal_order(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    effect = td["transition"]["effect"]
    swaps = [("after", "before"), ("before", "after"), ("then", "first"),
             ("next", "previously"), ("will", "already did"),
             ("reaches", "started at"), ("becomes", "was initially")]
    changed = False
    for a, b in swaps:
        if a in effect.lower():

            idx = effect.lower().index(a)
            effect = effect[:idx] + b + effect[idx + len(a):]
            changed = True
            break

    if not changed:

        effect = f"Before the interaction, {effect.lower()}"
        changed = True


    evidence = td["transition"].get("evidence", [])
    td["transition"]["evidence"] = [
        e.replace("initial", "FINAL").replace("final", "initial").replace("FINAL", "final")
        for e in evidence
    ]
    td["transition"]["effect"] = effect
    return td, "Reversed temporal order in transition"


@_register("unit_scale", "state_0", "seen", "wrong_unit")
def wrong_unit(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)

    forces = td["state_0"]["forces"]
    for f in forces:
        if f.get("unit") in _UNIT_CONFUSIONS:
            old = f["unit"]
            f["unit"] = _UNIT_CONFUSIONS[old]
            return td, f"Changed unit on '{f['name']}': {old} → {f['unit']}"

    if td["answer"].get("unit") in _UNIT_CONFUSIONS:
        old = td["answer"]["unit"]
        td["answer"]["unit"] = _UNIT_CONFUSIONS[old]
        return td, f"Changed answer unit: {old} → {td['answer']['unit']}"
    return td, "no units to change"


@_register("unit_scale", "state_0", "seen", "scale_magnitude_error")
def scale_magnitude_error(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    forces = td["state_0"]["forces"]
    numeric = [f for f in forces if isinstance(f.get("magnitude"), (int, float))]
    if numeric:
        f = random.choice(numeric)
        factor = random.choice([0.01, 0.1, 10, 100])
        old = f["magnitude"]
        f["magnitude"] = round(old * factor, 4)
        return td, f"Scaled '{f['name']}' magnitude by {factor}: {old} → {f['magnitude']}"

    vs = td["state_0"]["variables"]
    numeric_v = [(k, v) for k, v in vs.items() if isinstance(v, (int, float)) and v != 0]
    if numeric_v:
        k, v = random.choice(numeric_v)
        factor = random.choice([0.01, 0.1, 10, 100])
        vs[k] = round(v * factor, 4)
        return td, f"Scaled variable '{k}' by {factor}: {v} → {vs[k]}"
    return td, "no magnitudes to scale"


@_register("unit_scale", "answer", "held_out", "wrong_answer_unit")
def wrong_answer_unit(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    unit = td["answer"].get("unit")
    if unit and unit in _UNIT_CONFUSIONS:
        old = unit
        td["answer"]["unit"] = _UNIT_CONFUSIONS[old]
        return td, f"Wrong answer unit: {old} → {td['answer']['unit']}"
    return td, "no answer unit to change"


@_register("faithfulness", "answer", "seen", "contradict_trace")
def contradict_trace(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old_val = td["answer"]["value"]
    if isinstance(old_val, (int, float)):
        td["answer"]["value"] = round(-old_val if old_val != 0 else 42, 2)
    elif isinstance(old_val, str):
        contradictions = {
            "floats": "sinks", "sinks": "floats",
            "yes": "no", "no": "yes",
            "A": "C", "B": "D", "C": "A", "D": "B",
        }
        td["answer"]["value"] = contradictions.get(old_val, f"not_{old_val}")
    return td, f"Answer contradicts trace: '{old_val}' → '{td['answer']['value']}'"


@_register("faithfulness", "answer", "seen", "random_multiple_choice")
def random_multiple_choice(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old = td["answer"]["value"]
    options = ["A", "B", "C", "D"]
    if str(old) in options:
        wrong = random.choice([o for o in options if o != str(old)])
    else:
        wrong = f"wrong_{old}"
    td["answer"]["value"] = wrong
    td["answer"]["explanation"] = f"[FAITHFULNESS ERROR] Trace implies {old} but answer says {wrong}"
    return td, f"Random wrong answer: '{old}' → '{wrong}'"


@_register("faithfulness", "answer", "held_out", "partial_contradiction")
def partial_contradiction(td: dict) -> tuple[dict, str]:
    td = _deep_copy(td)
    old_val = td["answer"]["value"]
    if isinstance(old_val, (int, float)):
        td["answer"]["value"] = round(old_val * random.choice([-1, 0.5, 2.0]), 2)
        return td, f"Partial contradiction: {old_val} → {td['answer']['value']}"
    return td, "non-numeric answer, no partial contradiction"


def get_perturbations(family: str | None = None, label: str | None = None) -> list[Perturbation]:
    result = _PERTURBATIONS
    if family:
        result = [p for p in result if p.family == family]
    if label:
        result = [p for p in result if p.label == label]
    return result


def perturb_trace(trace: Trace, perturbation: Perturbation | None = None) -> PreferencePair:
    if perturbation is None:
        perturbation = random.choice(_PERTURBATIONS)

    chosen_dict = trace.to_dict()
    rejected_dict, description = perturbation.apply(_deep_copy(chosen_dict))

    pair_id = f"pair_{trace.id}_{perturbation.name}_{uuid.uuid4().hex[:6]}"

    return PreferencePair(
        id=pair_id,
        source_trace_id=trace.id,
        chosen=chosen_dict,
        rejected=rejected_dict,
        perturbation_type=perturbation.label,
        perturbation_field=perturbation.field,
        perturbation_family=perturbation.family,
        description=description,
    )


def _is_noop(chosen: dict, rejected: dict) -> bool:
    return json.dumps(chosen, sort_keys=True) == json.dumps(rejected, sort_keys=True)


def _applicable_perturbations(trace: Trace) -> list[Perturbation]:
    td = trace.to_dict()
    s0 = td.get("state_0", {})
    n_objects = len(s0.get("objects", []))
    n_forces = len(s0.get("forces", []))
    n_numeric_forces = sum(1 for f in s0.get("forces", [])
                          if isinstance(f.get("magnitude"), (int, float)))
    n_relations = len(s0.get("relations", []))
    n_numeric_vars = sum(1 for v in s0.get("variables", {}).values()
                        if isinstance(v, (int, float)) and v != 0)
    has_equation = bool(td.get("transition", {}).get("equation", ""))
    has_answer_unit = bool(td.get("answer", {}).get("unit"))
    answer_val = td.get("answer", {}).get("value")
    answer_is_numeric = isinstance(answer_val, (int, float))


    effect = td.get("transition", {}).get("effect", "").lower()
    has_temporal = any(kw in effect for kw in
                       ["after", "before", "then", "next", "previously"])

    applicable = []
    for p in _PERTURBATIONS:
        ok = True
        name = p.name


        if name in ("reverse_force_direction", "remove_force") and n_forces == 0:
            ok = False
        if name == "swap_force_target" and (n_forces == 0 or n_objects < 2):
            ok = False
        if name == "double_force_magnitude" and n_numeric_forces == 0:
            ok = False


        if name in ("swap_relation_type", "reverse_relation_args") and n_relations == 0:
            ok = False
        if name == "add_phantom_relation" and n_objects < 2:
            ok = False


        if name == "merge_objects" and n_objects < 2:
            ok = False


        if name in ("flip_variable_sign", "zero_out_variable") and n_numeric_vars == 0:
            ok = False
        if name == "swap_two_variables" and n_numeric_vars < 2:
            ok = False


        if name == "corrupt_equation":
            eq = td.get("transition", {}).get("equation", "")
            if not eq or not any(op in eq for op in ["+", "-", "*", "/", "sin", "cos", "²"]):
                ok = False


        if name == "reverse_temporal_order" and not has_temporal:
            ok = False


        if name == "wrong_unit":
            has_applicable_unit = any(
                f.get("unit") in _UNIT_CONFUSIONS for f in s0.get("forces", [])
            ) or (has_answer_unit and td["answer"]["unit"] in _UNIT_CONFUSIONS)
            if not has_applicable_unit:
                ok = False
        if name == "wrong_answer_unit" and not (has_answer_unit and td["answer"].get("unit") in _UNIT_CONFUSIONS):
            ok = False


        if name == "partial_contradiction" and not answer_is_numeric:
            ok = False

        if ok:
            applicable.append(p)

    return applicable


def generate_preference_pairs(
    traces: list[Trace],
    pairs_per_trace: int = 16,
    balance_labels: bool = True,
    max_retries: int = 5,
) -> list[PreferencePair]:
    pairs = []
    noop_skipped = 0

    for trace in traces:
        applicable = _applicable_perturbations(trace)
        if not applicable:
            continue

        if balance_labels:

            by_label: dict[str, list[Perturbation]] = {}
            for p in applicable:
                by_label.setdefault(p.label, []).append(p)


            labels = list(by_label.keys())
            selected = []
            idx = 0
            while len(selected) < pairs_per_trace and labels:
                label = labels[idx % len(labels)]
                pool = by_label[label]
                selected.append(random.choice(pool))
                idx += 1

                if idx > pairs_per_trace * 3:
                    break
        else:
            selected = random.choices(applicable, k=pairs_per_trace)

        for pert in selected:

            for attempt in range(max_retries):
                pair = perturb_trace(trace, pert)
                if not _is_noop(pair.chosen, pair.rejected):
                    pairs.append(pair)
                    break

                alternatives = [p for p in applicable if p.label == pert.label and p is not pert]
                if alternatives:
                    pert = random.choice(alternatives)
            else:
                noop_skipped += 1

    if noop_skipped > 0:
        print(f"  [perturbation] Skipped {noop_skipped} no-op pairs after retries")

    return pairs


def perturbation_stats(pairs: list[PreferencePair]) -> dict:
    stats = {
        "total_pairs": len(pairs),
        "by_label": {},
        "by_family": {"seen": 0, "held_out": 0},
        "by_field": {},
        "by_perturbation_name": {},
    }
    for p in pairs:
        stats["by_label"][p.perturbation_type] = stats["by_label"].get(p.perturbation_type, 0) + 1
        stats["by_family"][p.perturbation_family] = stats["by_family"].get(p.perturbation_family, 0) + 1
        stats["by_field"][p.perturbation_field] = stats["by_field"].get(p.perturbation_field, 0) + 1
        desc_key = p.description.split(":")[0] if ":" in p.description else p.description[:40]
        stats["by_perturbation_name"][desc_key] = stats["by_perturbation_name"].get(desc_key, 0) + 1
    return stats
