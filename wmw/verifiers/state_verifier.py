from __future__ import annotations
from wmw.schemas.models import VerifierResult


_VARIABLE_BOUNDS = {
    "mass": (0, 1e6),
    "angle": (-360, 360),
    "velocity": (-3e8, 3e8),
    "speed": (0, 3e8),
    "acceleration": (-1e6, 1e6),
    "force": (-1e9, 1e9),
    "distance": (0, 1e12),
    "height": (-1e4, 1e6),
    "radius": (0, 1e12),
    "temperature": (-273.15, 1e8),
    "pressure": (0, 1e12),
    "frequency": (0, 1e15),
    "wavelength": (0, 1e12),
    "resistance": (0, 1e12),
    "voltage": (-1e6, 1e6),
    "current": (-1e6, 1e6),
    "density": (0, 1e6),
    "k_N_per_m": (0, 1e9),
    "mu": (0, 10),
    "g": (0, 100),
    "depth": (0, 1e6),
}


_CONTRADICTORY_RELATIONS = {
    ("above", "below"), ("below", "above"),
    ("left_of", "right_of"), ("right_of", "left_of"),
    ("contact", "separated"), ("separated", "contact"),
    ("on", "below"), ("below", "on"),
}


def verify_state(trace_dict: dict, gold: dict | None = None) -> VerifierResult:
    details: list[str] = []
    labels: list[str] = []
    s0 = trace_dict.get("state_0", {})

    object_names = {o["name"] for o in s0.get("objects", []) if isinstance(o, dict)}

    object_names_lower = {n.lower() for n in object_names}

    _GENERIC_TARGETS = {
        "ground", "surface", "incline", "wall", "pivot", "center",
        "environment", "earth", "floor", "table", "ceiling", "air",
        "spring", "rope", "string", "wire", "axis", "origin",
        "fulcrum", "hinge", "support", "track", "ramp", "slope",
        "fluid", "water", "liquid", "medium", "lens", "mirror",
        "battery", "source", "resistor", "capacitor", "coil",
        "system", "body", "particle", "point", "mass",
    }

    def _is_known_entity(name: str) -> bool:
        if not name:
            return True
        if name in object_names:
            return True
        if name.lower() in object_names_lower:
            return True
        if name.lower() in _GENERIC_TARGETS:
            return True

        if any(c in name for c in "·*+/=_"):
            return True

        for on in object_names:
            if on.lower() in name.lower() or name.lower() in on.lower():
                return True

        stripped = name.lower().strip().replace("the ", "")
        if stripped in object_names_lower:
            return True
        for on in object_names_lower:
            if on in stripped or stripped in on:
                return True
        return False


    for i, f in enumerate(s0.get("forces", [])):
        if not isinstance(f, dict):
            continue
        target = f.get("target", "")
        if not _is_known_entity(target):
            details.append(f"force[{i}] '{f.get('name','')}' targets '{target}' which is not in objects list")
            if "object" not in labels:
                labels.append("object")


    for i, r in enumerate(s0.get("relations", [])):
        if not isinstance(r, dict):
            continue
        args = r.get("args", [])
        for arg in args:
            if not _is_known_entity(arg):
                details.append(f"relation[{i}] references '{arg}' not in objects list")
                if "object" not in labels:
                    labels.append("object")


    rel_set = set()
    for r in s0.get("relations", []):
        if not isinstance(r, dict):
            continue
        args = tuple(r.get("args", []))
        rtype = r.get("type", "")
        if len(args) == 2:
            rel_set.add((rtype, args[0], args[1]))

    for rtype1, a1, b1 in rel_set:
        for rtype2, a2, b2 in rel_set:
            if (rtype1, rtype2) in _CONTRADICTORY_RELATIONS and a1 == a2 and b1 == b2:
                details.append(f"contradictory relations: {rtype1}({a1},{b1}) vs {rtype2}({a2},{b2})")
                if "relation" not in labels:
                    labels.append("relation")


    variables = s0.get("variables", {})
    for var_name, val in variables.items():
        if not isinstance(val, (int, float)):
            continue
        for pattern, (lo, hi) in _VARIABLE_BOUNDS.items():
            if pattern in var_name.lower():
                if val < lo or val > hi:
                    details.append(f"variable '{var_name}' = {val} outside plausible range [{lo}, {hi}]")
                    if "state" not in labels:
                        labels.append("state")
                break


    for f in s0.get("forces", []):
        if not isinstance(f, dict):
            continue
        fname = f.get("name", "").lower()
        direction = f.get("direction", "").lower()

        if "gravity" in fname and "down" not in direction and "toward" not in direction:
            details.append(f"gravity force has unexpected direction: '{direction}'")
            if "force" not in labels:
                labels.append("force")

        if "normal" in fname and "into" in direction:
            details.append(f"normal force points into surface: '{direction}'")
            if "force" not in labels:
                labels.append("force")


    if gold:
        gold_vars = gold.get("variables", {})
        for k, gv in gold_vars.items():
            if isinstance(gv, (int, float)) and k in variables:
                pv = variables[k]
                if isinstance(pv, (int, float)):
                    if gv != 0 and abs(pv - gv) / max(abs(gv), 1e-9) > 0.01:
                        details.append(f"variable '{k}': predicted {pv}, gold {gv}")
                        if "state" not in labels:
                            labels.append("state")
                    elif gv == 0 and abs(pv) > 1e-6:
                        details.append(f"variable '{k}': predicted {pv}, gold 0")
                        if "state" not in labels:
                            labels.append("state")

    state_ok = len(labels) == 0
    if not details:
        details.append("state: all checks passed")

    return VerifierResult(
        state_ok=state_ok,
        details=details,
        labels=labels,
    )
