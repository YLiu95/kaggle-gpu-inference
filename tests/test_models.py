from pathlib import Path
import struct

import pytest

from kaggle_gpu_inference.models import context_estimates, load_gguf_config, parse_model_ref


MODEL_URL = (
    "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/"
    "blob/main/Qwen3.6-35B-A3B-UD-IQ1_M.gguf"
)


def test_parse_hugging_face_blob_url() -> None:
    ref = parse_model_ref(MODEL_URL)
    assert ref.repo_id == "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    assert ref.revision == "main"
    assert ref.filename == "Qwen3.6-35B-A3B-UD-IQ1_M.gguf"


def test_parse_repo_with_filename() -> None:
    ref = parse_model_ref("owner/repository", "model.gguf")
    assert ref.repo_id == "owner/repository"
    assert ref.filename == "model.gguf"


def test_parse_local_model(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.touch()
    assert parse_model_ref(str(model)).repo_id == "local"


def test_reject_ambiguous_name() -> None:
    with pytest.raises(ValueError):
        parse_model_ref("model")


def test_context_estimate_respects_model_limit() -> None:
    config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 32768,
    }
    one, two = context_estimates(8_000_000_000, config, 16_000_000_000)
    assert 0 < one <= 32768
    assert one <= two <= 32768


def test_context_estimate_returns_zero_when_weights_do_not_fit() -> None:
    config = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
    }
    assert context_estimates(20_000_000_000, config, 16_000_000_000)[0] == 0


def test_load_gguf_config(tmp_path: Path) -> None:
    values = {
        "general.architecture": (8, "llama"),
        "llama.block_count": (4, 32),
        "llama.attention.head_count": (4, 32),
        "llama.attention.head_count_kv": (4, 8),
        "llama.embedding_length": (4, 4096),
        "llama.context_length": (4, 32768),
    }
    path = tmp_path / "metadata.gguf"
    with path.open("wb") as handle:
        handle.write(b"GGUF" + struct.pack("<IQQ", 3, 0, len(values)))
        for key, (value_type, value) in values.items():
            encoded_key = key.encode()
            handle.write(struct.pack("<Q", len(encoded_key)) + encoded_key + struct.pack("<I", value_type))
            if value_type == 8:
                encoded_value = value.encode()
                handle.write(struct.pack("<Q", len(encoded_value)) + encoded_value)
            else:
                handle.write(struct.pack("<I", value))

    assert load_gguf_config(str(path)) == {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 32768,
    }
