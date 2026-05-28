from __future__ import annotations
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    name: str
    provider: str
    model_id: str
    api_key_env: str = ""
    base_url: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    timeout: int = 120
    rate_limit_rpm: int = 30


MODELS = {
    "gpt4o": ModelConfig(
        name="GPT-4o",
        provider="openai",
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    ),
    "gpt4o_mini": ModelConfig(
        name="GPT-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    ),
    "gpt55": ModelConfig(
        name="GPT-5.5",
        provider="openai",
        model_id="gpt-5.5",
        api_key_env="OPENAI_API_KEY",
    ),
    "gemini_flash": ModelConfig(
        name="Gemini 3.5 Flash",
        provider="openai",
        model_id="gemini-3.5-flash",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    ),
    "claude_sonnet": ModelConfig(
        name="Claude Sonnet 4",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "claude_opus": ModelConfig(
        name="Claude Opus 4.7",
        provider="anthropic",
        model_id="claude-opus-4-7",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "claude_haiku": ModelConfig(
        name="Claude Haiku 3.5",
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "llava_local": ModelConfig(
        name="LLaVA-1.6",
        provider="local",
        model_id="llava-v1.6-vicuna-13b",
        base_url="http://localhost:8000/v1",
    ),
    "qwen_vl": ModelConfig(
        name="Qwen-VL-Plus",
        provider="local",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        base_url="http://localhost:8000/v1",
    ),
    "mock": ModelConfig(
        name="Mock (testing)",
        provider="mock",
        model_id="mock-v1",
    ),
}


@dataclass
class VLMResponse:
    raw_text: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


def _encode_image_base64(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, media_type


def _call_openai(
    config: ModelConfig,
    system: str,
    user_text: str,
    image_path: str | None = None,
) -> VLMResponse:
    import urllib.request

    api_key = os.environ.get(config.api_key_env, "")


    if not api_key:
        if config.provider == "local":
            api_key = "EMPTY"
        else:
            return VLMResponse(raw_text="", model=config.model_id, latency_ms=0,
                               error=f"Missing API key: set {config.api_key_env}")

    base_url = config.base_url or "https://api.openai.com/v1"


    content = []
    if image_path and Path(image_path).exists():
        img_data, media_type = _encode_image_base64(image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{img_data}"}
        })
    content.append({"type": "text", "text": user_text})

    payload = {
        "model": config.model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
    }


    if config.model_id.startswith(("gpt-5", "gpt-4.1")):
        payload["max_completion_tokens"] = config.max_tokens

    else:
        payload["max_tokens"] = config.max_tokens
        payload["temperature"] = config.temperature

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read())

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return VLMResponse(
            raw_text=text,
            model=config.model_id,
            latency_ms=(time.time() - t0) * 1000,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )
    except Exception as e:
        return VLMResponse(
            raw_text="", model=config.model_id,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


def _call_anthropic(
    config: ModelConfig,
    system: str,
    user_text: str,
    image_path: str | None = None,
) -> VLMResponse:
    import urllib.request

    api_key = os.environ.get(config.api_key_env, "")
    if not api_key:
        return VLMResponse(raw_text="", model=config.model_id, latency_ms=0,
                           error=f"Missing API key: set {config.api_key_env}")


    content = []
    if image_path and Path(image_path).exists():
        img_data, media_type = _encode_image_base64(image_path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_data},
        })
    content.append({"type": "text", "text": user_text})

    payload = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read())

        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return VLMResponse(
            raw_text=text,
            model=config.model_id,
            latency_ms=(time.time() - t0) * 1000,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )
    except Exception as e:
        return VLMResponse(
            raw_text="", model=config.model_id,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


def _call_mock(
    config: ModelConfig,
    system: str,
    user_text: str,
    image_path: str | None = None,
) -> VLMResponse:
    import random
    mock_trace = {
        "state_0": {
            "objects": [{"name": "object_A", "attributes": {"mass": round(random.uniform(1, 10), 1)}}],
            "relations": [{"type": "on", "args": ["object_A", "surface"]}],
            "forces": [{"name": "gravity", "target": "object_A", "direction": "downward",
                        "magnitude": round(random.uniform(5, 100), 1), "unit": "N"}],
            "variables": {"v0": round(random.uniform(0, 20), 1)},
            "assumptions": ["no air resistance"],
        },
        "transition": {
            "rule": "Newton's second law",
            "effect": f"Object accelerates at {round(random.uniform(1, 10), 1)} m/s²",
            "equation": "F = ma",
            "evidence": ["net force applied"],
        },
        "state_1": {
            "predicted_change": f"Velocity changes to {round(random.uniform(1, 30), 1)} m/s",
            "new_variables": {"v_final": round(random.uniform(1, 30), 1)},
        },
        "answer": {
            "value": round(random.uniform(1, 50), 2),
            "unit": "m/s",
            "explanation": "Computed from F=ma",
        },
    }
    return VLMResponse(
        raw_text=json.dumps(mock_trace),
        model="mock-v1",
        latency_ms=random.uniform(10, 50),
        input_tokens=100,
        output_tokens=200,
    )


_CALLERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "local": _call_openai,
    "mock": _call_mock,
}


def call_vlm(
    config: ModelConfig,
    system: str,
    user_text: str,
    image_path: str | None = None,
) -> VLMResponse:
    caller = _CALLERS.get(config.provider)
    if caller is None:
        return VLMResponse(
            raw_text="", model=config.model_id, latency_ms=0,
            error=f"Unknown provider: {config.provider}",
        )
    return caller(config, system, user_text, image_path)
