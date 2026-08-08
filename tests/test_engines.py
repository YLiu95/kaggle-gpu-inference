import json
import os

import psutil
import pytest

from kaggle_gpu_inference import engines


class FakeResponse:
    status = 200

    def __init__(self, lines=None, payload=None):
        self.lines = lines or []
        self.payload = payload or {"tokens": [1, 2]}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self, *args):
        return json.dumps(self.payload).encode()


def test_state_process_running_rejects_reused_pid() -> None:
    process = psutil.Process(os.getpid())
    assert engines._state_process_running({
        "pid": process.pid,
        "process_create_time": process.create_time(),
    })
    assert not engines._state_process_running({
        "pid": process.pid,
        "process_create_time": process.create_time() - 100,
    })


def test_llama_server_command_enables_mtp(monkeypatch) -> None:
    monkeypatch.setattr(engines.Path, "exists", lambda self: True)
    command = engines._server_command("llama.cpp", "/model.gguf", 1, 8192, "draft-mtp", 2)
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert command[command.index("--spec-draft-n-max") + 1] == "2"


def test_llama_server_command_omits_disabled_speculation(monkeypatch) -> None:
    monkeypatch.setattr(engines.Path, "exists", lambda self: True)
    command = engines._server_command("llama.cpp", "/model.gguf", 1, 8192)
    assert "--spec-type" not in command
    assert "--spec-draft-n-max" not in command


def test_vllm_tpu_command_uses_all_chips() -> None:
    command = engines._server_command("vllm", "owner/model", 8, 32768, accelerator_kind="tpu")
    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert command[command.index("--max-model-len") + 1] == "32768"


def test_sglang_tpu_command_uses_jax_backend() -> None:
    command = engines._server_command("sglang", "owner/model", 8, 32768, accelerator_kind="tpu")
    assert "sgl_jax.launch_server" in command
    assert command[command.index("--tp-size") + 1] == "8"
    assert command[command.index("--device") + 1] == "tpu"


def test_llama_rejects_tpu() -> None:
    with pytest.raises(RuntimeError, match="only on GPU"):
        engines._server_command("llama.cpp", "/model.gguf", 8, 4096, accelerator_kind="tpu")


def test_stream_completion_sends_requested_model(monkeypatch) -> None:
    captured = {}

    def fake_open(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse([
            b'data: {"choices":[{"delta":{"content":"OK"}}]}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert list(engines.stream_completion("owner/model", "prompt", 4, 0.1)) == [
        engines.StreamChunk(content="OK")
    ]
    assert captured["model"] == "owner/model"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["stream_options"] == {"include_usage": True}


def test_stream_completion_reports_final_usage(monkeypatch) -> None:
    def fake_open(request, timeout):
        return FakeResponse([
            b'data: {"choices":[],"usage":{"completion_tokens":7}}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert list(engines.stream_completion("owner/model", "prompt", 8, 0.1)) == [
        engines.StreamChunk(output_tokens=7)
    ]


def test_stream_completion_accepts_reasoning_content(monkeypatch) -> None:
    captured = {}

    def fake_open(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse([
            b'data: {"choices":[{"delta":{"reasoning_content":"Think"}}]}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert list(engines.stream_completion("owner/model", "prompt", 4, 0.1, thinking=True)) == [
        engines.StreamChunk(reasoning="Think")
    ]
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}


def test_stream_completion_keeps_mixed_delta_as_one_event(monkeypatch) -> None:
    def fake_open(request, timeout):
        return FakeResponse([
            b'data: {"choices":[{"delta":{"reasoning_content":"Think","content":"Answer"}}]}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert list(engines.stream_completion("owner/model", "prompt", 4, 0.1, thinking=True)) == [
        engines.StreamChunk(reasoning="Think", content="Answer")
    ]


def test_token_count_uses_engine_specific_schema(monkeypatch) -> None:
    payloads = []

    def fake_open(request, timeout):
        payloads.append(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert engines.token_count("llama.cpp", "/model.gguf", "hello") == 2
    assert engines.token_count("vllm", "owner/model", "hello") == 2
    assert engines.token_count("sglang", "owner/model", "hello") == 2
    assert payloads == [
        {"content": "hello"},
        {"model": "owner/model", "prompt": "hello"},
        {"text": "hello"},
    ]


def test_token_count_uses_chat_schema_for_vllm(monkeypatch) -> None:
    captured = {}

    def fake_open(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert engines.token_count("vllm", "owner/model", "hello", chat=True) == 2
    assert captured == {
        "model": "owner/model",
        "messages": [{"role": "user", "content": "hello"}],
    }