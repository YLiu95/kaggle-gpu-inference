import json
import os

import psutil

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


def test_stream_completion_sends_requested_model(monkeypatch) -> None:
    captured = {}

    def fake_open(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse([
            b'data: {"choices":[{"delta":{"content":"OK"}}]}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert list(engines.stream_completion("owner/model", "prompt", 4, 0.1)) == ["OK"]
    assert captured["model"] == "owner/model"
    assert captured["chat_template_kwargs"] == {"enable_thinking": False}


def test_stream_completion_accepts_reasoning_content(monkeypatch) -> None:
    captured = {}

    def fake_open(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse([
            b'data: {"choices":[{"delta":{"reasoning_content":"Think"}}]}\n',
            b'data: [DONE]\n',
        ])

    monkeypatch.setattr(engines.urllib.request, "urlopen", fake_open)
    assert list(engines.stream_completion("owner/model", "prompt", 4, 0.1, thinking=True)) == ["Think"]
    assert captured["chat_template_kwargs"] == {"enable_thinking": True}


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