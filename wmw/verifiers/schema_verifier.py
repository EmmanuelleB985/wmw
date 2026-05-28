from __future__ import annotations
import json
import os
from pathlib import Path

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from wmw.schemas.models import VerifierResult


_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "trace_schema.json"
_SCHEMA: dict | None = None


def _load_schema() -> dict:
    global _SCHEMA
    if _SCHEMA is None:
        with open(_SCHEMA_PATH) as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


_REQUIRED_TOP = {"id", "scenario_family", "state_0", "transition", "state_1", "answer", "metadata"}
_REQUIRED_STATE = {"objects", "relations", "forces", "variables"}
_REQUIRED_TRANSITION = {"rule", "effect"}
_REQUIRED_STATE1 = {"predicted_change"}
_REQUIRED_ANSWER = {"value"}

_VALID_FAMILIES = {
    "inclined_plane", "projectile", "collision", "pulley",
    "spring", "circuit", "fluid", "thermal",
    "free_fall", "friction", "circular_motion", "wave",
    "lever", "buoyancy", "optics", "pendulum",
    "em_induction", "unknown",
}

_VALID_RELATION_TYPES = {
    "on", "in", "attached", "adjacent", "above", "below",
    "left_of", "right_of", "contact", "supported_by",
    "connected", "separated", "aligned", "perpendicular",
}

_VALID_LABELS = {
    "object", "state", "relation", "force",
    "transition", "intervention", "temporal",
    "unit_scale", "faithfulness",
}


def verify_schema(trace_dict: dict) -> VerifierResult:
    details: list[str] = []
    labels: list[str] = []


    source = trace_dict.get("metadata", {}).get("source", "")
    if HAS_JSONSCHEMA and source != "model_generated":
        schema = _load_schema()
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(trace_dict), key=lambda e: list(e.path))
        for err in errors:
            path = ".".join(str(p) for p in err.absolute_path) or "(root)"
            details.append(f"schema: {path}: {err.message}")
        if errors:
            structural = [d for d in details if "missing" in d.lower() or "required" in d.lower()]
            return VerifierResult(
                schema_ok=False,
                details=details[:20],
                labels=["object"] if structural else [],
            )
        return VerifierResult(schema_ok=True, details=["schema: all checks passed"])


    missing_top = _REQUIRED_TOP - set(trace_dict.keys())
    if missing_top:
        details.append(f"missing top-level fields: {missing_top}")


    fam = trace_dict.get("scenario_family")
    source = trace_dict.get("metadata", {}).get("source", "")
    if fam and fam not in _VALID_FAMILIES and source != "model_generated":
        details.append(f"unknown scenario_family: '{fam}'")


    s0 = trace_dict.get("state_0", {})
    if isinstance(s0, dict):
        missing_s0 = _REQUIRED_STATE - set(s0.keys())
        if missing_s0:
            details.append(f"state_0 missing: {missing_s0}")

        for i, obj in enumerate(s0.get("objects", [])):
            if not isinstance(obj, dict) or "name" not in obj:
                details.append(f"state_0.objects[{i}]: must be dict with 'name'")

        for i, rel in enumerate(s0.get("relations", [])):
            if not isinstance(rel, dict):
                details.append(f"state_0.relations[{i}]: must be dict")
            elif rel.get("type") not in _VALID_RELATION_TYPES:
                details.append(f"state_0.relations[{i}]: unknown type '{rel.get('type')}'")
            elif not isinstance(rel.get("args"), list) or len(rel.get("args", [])) != 2:
                details.append(f"state_0.relations[{i}]: args must be [str, str]")

        for i, f in enumerate(s0.get("forces", [])):
            if not isinstance(f, dict):
                details.append(f"state_0.forces[{i}]: must be dict")
            else:
                for req in ("name", "target", "direction"):
                    if req not in f:
                        details.append(f"state_0.forces[{i}]: missing '{req}'")
    else:
        details.append("state_0: must be a dict")


    tr = trace_dict.get("transition", {})
    if isinstance(tr, dict):
        missing_tr = _REQUIRED_TRANSITION - set(tr.keys())
        if missing_tr:
            details.append(f"transition missing: {missing_tr}")
    else:
        details.append("transition: must be a dict")


    s1 = trace_dict.get("state_1", {})
    if isinstance(s1, dict):
        missing_s1 = _REQUIRED_STATE1 - set(s1.keys())
        if missing_s1:
            details.append(f"state_1 missing: {missing_s1}")
    else:
        details.append("state_1: must be a dict")


    ans = trace_dict.get("answer", {})
    if isinstance(ans, dict):
        if "value" not in ans:
            details.append("answer missing 'value'")
    else:
        details.append("answer: must be a dict")


    ver = trace_dict.get("verifier", {})
    if isinstance(ver, dict) and "labels" in ver:
        for lbl in ver["labels"]:
            if lbl not in _VALID_LABELS:
                details.append(f"verifier.labels: unknown label '{lbl}'")

    schema_ok = len(details) == 0
    if not schema_ok:

        structural_failures = [d for d in details if "missing" in d or "must be" in d]
        if structural_failures:
            labels.append("object")

    return VerifierResult(
        schema_ok=schema_ok,
        details=details if details else ["schema: all checks passed"],
        labels=labels,
    )
