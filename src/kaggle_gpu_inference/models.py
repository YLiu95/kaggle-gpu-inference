from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import HfApi, hf_hub_download

from .config import MODELS_DIR, ensure_dirs, hf_token


@dataclass(frozen=True)
class ModelRef:
    source: str
    repo_id: str
    revision: str = "main"
    filename: str | None = None

    @property
    def name(self) -> str:
        return Path(self.filename).name if self.filename else self.repo_id


def parse_model_ref(value: str, filename: str | None = None) -> ModelRef:
    if value.startswith(("https://huggingface.co/", "http://huggingface.co/")):
        parts = [unquote(part) for part in urlparse(value).path.strip("/").split("/")]
        if len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
            return ModelRef(value, "/".join(parts[:2]), parts[3], "/".join(parts[4:]))
        if len(parts) >= 2:
            return ModelRef(value, "/".join(parts[:2]), filename=filename)
        raise ValueError(f"Invalid Hugging Face URL: {value}")
    path = Path(value).expanduser()
    if path.exists():
        return ModelRef(str(path.resolve()), "local", filename=str(path.resolve()))
    if "/" in value:
        return ModelRef(value, value, filename=filename)
    raise ValueError("Model must be a local path, owner/repository, or Hugging Face URL")


def resolve_model(ref: ModelRef, engine: str) -> str:
    if ref.repo_id == "local":
        return ref.filename or ref.source
    if engine != "llama.cpp":
        return ref.repo_id
    if not ref.filename:
        raise ValueError("llama.cpp requires a GGUF file URL or --filename")
    if not ref.filename.lower().endswith(".gguf"):
        raise ValueError("llama.cpp requires a .gguf model file")
    ensure_dirs()
    local_dir = MODELS_DIR / ref.repo_id.replace("/", "--")
    return hf_hub_download(
        repo_id=ref.repo_id,
        filename=ref.filename,
        revision=ref.revision,
        local_dir=local_dir,
        token=hf_token(),
    )


def model_size_bytes(model: str) -> int:
    path = Path(model)
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def estimated_model_size_bytes(ref: ModelRef, resolved_model: str) -> int:
    local_size = model_size_bytes(resolved_model)
    if local_size or ref.repo_id == "local":
        return local_size
    try:
        siblings = HfApi(token=hf_token()).model_info(
            ref.repo_id, revision=ref.revision, files_metadata=True
        ).siblings
        safetensors = [item for item in siblings if item.rfilename.endswith(".safetensors")]
        weights = safetensors or [item for item in siblings if item.rfilename.endswith((".bin", ".pt"))]
        return sum(int(item.size or 0) for item in weights)
    except Exception:
        return 0


def load_hf_config(ref: ModelRef) -> dict:
    if ref.repo_id == "local":
        return {}
    try:
        path = hf_hub_download(
            repo_id=ref.repo_id,
            filename="config.json",
            revision=ref.revision,
            local_dir=MODELS_DIR / ref.repo_id.replace("/", "--"),
            token=hf_token(),
        )
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def load_gguf_config(model: str) -> dict:
    path = Path(model)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return {}
    scalar_formats = {
        0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
        6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
    }

    def read_string(handle) -> str:
        length = struct.unpack("<Q", handle.read(8))[0]
        return handle.read(length).decode("utf-8", errors="replace")

    def read_value(handle, value_type: int):
        if value_type == 8:
            return read_string(handle)
        if value_type in scalar_formats:
            value_format = scalar_formats[value_type]
            return struct.unpack(f"<{value_format}", handle.read(struct.calcsize(value_format)))[0]
        if value_type == 9:
            item_type = struct.unpack("<I", handle.read(4))[0]
            item_count = struct.unpack("<Q", handle.read(8))[0]
            if item_type in scalar_formats:
                handle.seek(struct.calcsize(scalar_formats[item_type]) * item_count, os.SEEK_CUR)
            else:
                for _ in range(item_count):
                    read_value(handle, item_type)
            return None
        raise ValueError(f"Unsupported GGUF metadata type: {value_type}")

    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return {}
            version = struct.unpack("<I", handle.read(4))[0]
            if version not in {2, 3}:
                return {}
            _, metadata_count = struct.unpack("<QQ", handle.read(16))
            values: dict[str, object] = {}
            architecture = ""
            for _ in range(metadata_count):
                key = read_string(handle)
                value_type = struct.unpack("<I", handle.read(4))[0]
                value = read_value(handle, value_type)
                if key == "general.architecture":
                    architecture = str(value)
                    values[key] = value
                elif architecture and key.startswith(f"{architecture}."):
                    values[key] = value
            if not architecture:
                return {}
            prefix = f"{architecture}."
            return {
                "num_hidden_layers": values.get(prefix + "block_count", 0),
                "num_attention_heads": values.get(prefix + "attention.head_count", 0),
                "num_key_value_heads": values.get(prefix + "attention.head_count_kv", 0),
                "hidden_size": values.get(prefix + "embedding_length", 0),
                "max_position_embeddings": values.get(prefix + "context_length", 0),
            }
    except (EOFError, OSError, struct.error, ValueError):
        return {}


def context_estimates(model_bytes: int, config: dict, vram_per_gpu: int) -> tuple[int, int]:
    layers = int(config.get("num_hidden_layers", 0))
    heads = int(config.get("num_attention_heads", 0))
    kv_heads = int(config.get("num_key_value_heads", heads))
    hidden = int(config.get("hidden_size", 0))
    model_limit = int(config.get("max_position_embeddings", 0))
    if not all((layers, heads, kv_heads, hidden)):
        return (0, 0)
    kv_bytes_per_token = 2 * layers * kv_heads * (hidden // heads) * 2

    def estimate(gpus: int) -> int:
        usable = int(vram_per_gpu * gpus * 0.90) - model_bytes
        if usable <= 0:
            return 0
        value = usable // kv_bytes_per_token
        return min(value, model_limit) if model_limit else value

    return estimate(1), estimate(2)
