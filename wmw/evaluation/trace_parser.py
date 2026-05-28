from __future__ import annotations
import json
import re
from typing import Any


def parse_trace(raw_text: str) -> tuple[dict | None, str]:
    text = raw_text.strip()
    if not text:
        return None, "failed"


    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return d, "ok"
    except json.JSONDecodeError:
        pass


    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        try:
            d = json.loads(fence_match.group(1))
            if isinstance(d, dict):
                return d, "json_fence"
        except json.JSONDecodeError:
            pass


    brace_start = text.find('{')
    if brace_start >= 0:

        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[brace_start:i+1])
                        if isinstance(d, dict):
                            return d, "extracted"
                    except json.JSONDecodeError:
                        pass
                    break


    answer = _extract_answer_fallback(text)
    if answer is not None:
        return {"answer": {"value": answer}}, "answer_only"

    return None, "failed"


def _extract_answer_fallback(text: str) -> str | float | None:
    text = text.strip()


    if re.match(r'^[A-D]$', text):
        return text


    m = re.search(r'(?:the answer is|answer:?)\s*[:\-]?\s*([A-D]|\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        val = m.group(1)
        try:
            return float(val)
        except ValueError:
            return val


    m = re.match(r'^-?\d+\.?\d*$', text)
    if m:
        return float(text)

    return None


def extract_answer(trace_dict: dict | None) -> Any:
    if trace_dict is None:
        return None

    ans = trace_dict.get("answer", {})
    if isinstance(ans, dict):
        return ans.get("value")
    return ans


def normalize_answer(answer: Any) -> str:
    if answer is None:
        return ""
    s = str(answer).strip().lower()

    s = s.replace("option ", "").replace("(", "").replace(")", "")
    s = s.strip(".")

    try:
        f = float(s)
        return f"{f:.4g}"
    except (ValueError, TypeError):
        pass
    return s


def answers_match(predicted: Any, gold: Any, tolerance: float = 0.05,
                  options: list[str] | None = None) -> bool:
    p = normalize_answer(predicted)
    g = normalize_answer(gold)

    if not p or not g:
        return False


    if p == g:
        return True


    try:
        pf = float(p)
        gf = float(g)
        if gf == 0:
            return abs(pf) < tolerance
        return abs(pf - gf) / max(abs(gf), 1e-9) < tolerance
    except (ValueError, TypeError):
        pass


    if options and len(p) == 1 and p.isalpha():
        letter_idx = ord(p) - ord('a')
        if 0 <= letter_idx < len(options):
            resolved = normalize_answer(options[letter_idx])
            if resolved == g:
                return True


    if options and len(g) == 1 and g.isalpha():
        letter_idx = ord(g) - ord('a')
        if 0 <= letter_idx < len(options):
            resolved = normalize_answer(options[letter_idx])
            if resolved == p:
                return True


    if len(g) > 3 and g in p:
        return True
    if len(p) > 3 and p in g:
        return True

    return False
