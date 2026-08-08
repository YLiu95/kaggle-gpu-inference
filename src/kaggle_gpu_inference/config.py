from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


HOME = Path(os.environ.get("KGPU_HOME", "/root/kaggle-gpu-inference"))
MODELS_DIR = HOME / "models"
RUNTIME_DIR = HOME / "runtime"
LOG_DIR = HOME / "logs"
STATE_FILE = RUNTIME_DIR / "server.json"
PORT = int(os.environ.get("KGPU_PORT", "8088"))


def ensure_dirs() -> None:
    for path in (MODELS_DIR, RUNTIME_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def read_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_state(state: dict[str, Any]) -> None:
    ensure_dirs()
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(STATE_FILE)


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None
