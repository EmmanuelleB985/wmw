from __future__ import annotations
import os
import time
import json
import base64
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

from wmw.evaluation.vlm_caller import ModelConfig, VLMResponse


OPEN_VLM_REGISTRY: dict[str, dict] = {
    "qwen25_vl_7b": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "display_name": "Qwen2.5-VL-7B",
        "tensor_parallel": 1,
        "max_model_len": 8192,
        "size_class": "small",
    },
    "qwen25_vl_32b": {
        "hf_id": "Qwen/Qwen2.5-VL-32B-Instruct",
        "display_name": "Qwen2.5-VL-32B",
        "tensor_parallel": 2,
        "max_model_len": 8192,
        "size_class": "mid",
    },
    "qwen25_vl_72b": {
        "hf_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "display_name": "Qwen2.5-VL-72B",
        "tensor_parallel": 4,
        "max_model_len": 8192,
        "size_class": "large",
    },
    "internvl3_8b": {
        "hf_id": "OpenGVLab/InternVL3-8B",
        "display_name": "InternVL3-8B",
        "tensor_parallel": 1,
        "max_model_len": 8192,
        "size_class": "small",
    },
    "internvl3_14b": {
        "hf_id": "OpenGVLab/InternVL3-14B",
        "display_name": "InternVL3-14B",
        "tensor_parallel": 2,
        "max_model_len": 8192,
        "size_class": "mid",
    },
    "internvl3_38b": {
        "hf_id": "OpenGVLab/InternVL3-38B",
        "display_name": "InternVL3-38B",
        "tensor_parallel": 4,
        "max_model_len": 8192,
        "size_class": "large",
    },
    "llava_onevision_7b": {
        "hf_id": "lmms-lab/llava-onevision-qwen2-7b-ov",
        "display_name": "LLaVA-OneVision-7B",
        "tensor_parallel": 1,
        "max_model_len": 8192,
        "size_class": "small",
    },
    "molmo_7b": {
        "hf_id": "allenai/Molmo-7B-D-0924",
        "display_name": "Molmo-7B-D",
        "tensor_parallel": 1,
        "max_model_len": 4096,
        "size_class": "small",
    },
}


def open_vlm_config(key: str, base_url: str | None = None) -> ModelConfig:
    if key not in OPEN_VLM_REGISTRY:
        raise KeyError(f"Unknown open VLM '{key}'. Options: {list(OPEN_VLM_REGISTRY)}")
    spec = OPEN_VLM_REGISTRY[key]
    url = base_url or os.environ.get("OPEN_VLM_URL", "http://localhost:8000/v1")
    return ModelConfig(
        name=spec["display_name"],
        provider="local",
        model_id=spec["hf_id"],
        base_url=url,
        api_key_env="OPEN_VLM_KEY",
        max_tokens=2048,
        temperature=0.0,
        rate_limit_rpm=1000,
        timeout=180,
    )


def wait_for_server(base_url: str, timeout_s: int = 600) -> bool:
    url = base_url.rstrip("/") + "/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(5)
    return False


def check_server_model(base_url: str) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        return []


def ensure_local_key() -> None:
    if not os.environ.get("OPEN_VLM_KEY"):
        os.environ["OPEN_VLM_KEY"] = "EMPTY"
