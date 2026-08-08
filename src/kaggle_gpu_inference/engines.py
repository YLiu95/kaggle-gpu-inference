from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

import psutil

from .config import HOME, PORT, RUNTIME_DIR, STATE_FILE, ensure_dirs, hf_token, read_state, write_state


ENGINE_NAMES = ("llama.cpp", "vllm", "sglang")


def _server_command(engine: str, model: str, gpus: int, context: int) -> list[str]:
    if engine == "llama.cpp":
        binary = HOME / "vendor/llama.cpp/build/bin/llama-server"
        if not binary.exists():
            raise RuntimeError("llama.cpp is not installed; run: kgpu setup --engine llama.cpp")
        command = [
            str(binary), "--model", model, "--ctx-size", str(context),
            "--n-gpu-layers", "999", "--parallel", "1", "--host", "127.0.0.1",
            "--port", str(PORT), "--metrics",
        ]
        if gpus == 2:
            command.extend(["--split-mode", "layer", "--tensor-split", "1,1"])
        return command
    if engine == "vllm":
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
            "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", str(context),
            "--tensor-parallel-size", str(gpus), "--gpu-memory-utilization", "0.90",
        ]
    if engine == "sglang":
        return [
            sys.executable, "-m", "sglang.launch_server", "--model-path", model,
            "--host", "127.0.0.1", "--port", str(PORT), "--context-length", str(context),
            "--tp", str(gpus),
        ]
    raise ValueError(f"Unsupported engine: {engine}")


def _state_process_running(state: dict) -> bool:
    pid = state.get("pid")
    started = state.get("process_create_time")
    if not pid or started is None:
        return False
    try:
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - float(started)) < 0.1
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def _endpoint_healthy() -> bool:
    for url in (f"http://127.0.0.1:{PORT}/health", f"http://127.0.0.1:{PORT}/v1/models"):
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
    return False


def _stop_group(pid: int, force: bool = False) -> None:
    try:
        os.killpg(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def server_matches(engine: str, model: str, source: str, gpus: int, context: int) -> bool:
    expected = {"engine": engine, "model": model, "source": source, "gpus": gpus, "context": context}
    state = read_state()
    return (
        _state_process_running(state)
        and all(state.get(key) == value for key, value in expected.items())
        and _endpoint_healthy()
    )


def clear_server() -> bool:
    state = read_state()
    pid = state.get("pid")
    stopped = False
    if _state_process_running(state):
        _stop_group(pid)
        stopped = True
        deadline = time.monotonic() + 15
        while _state_process_running(state) and time.monotonic() < deadline:
            time.sleep(0.25)
        if _state_process_running(state):
            _stop_group(pid, force=True)
    STATE_FILE.unlink(missing_ok=True)
    return stopped


def ensure_server(engine: str, model: str, source: str, gpus: int, context: int) -> tuple[dict, bool]:
    ensure_dirs()
    signature = {"engine": engine, "model": model, "source": source, "gpus": gpus, "context": context}
    state = read_state()
    if (
        _state_process_running(state)
        and all(state.get(key) == value for key, value in signature.items())
        and _endpoint_healthy()
    ):
        return state, True
    clear_server()
    command = _server_command(engine, model, gpus, context)
    log_path = RUNTIME_DIR / "server.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1" if gpus == 2 else "0"
    if token := hf_token():
        environment["HF_TOKEN"] = token
    log_handle = log_path.open("w")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
    )
    log_handle.close()
    state = {
        **signature,
        "pid": process.pid,
        "process_create_time": psutil.Process(process.pid).create_time(),
        "port": PORT,
        "log": str(log_path),
    }
    write_state(state)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
            _stop_group(process.pid, force=True)
            STATE_FILE.unlink(missing_ok=True)
            raise RuntimeError(f"{engine} server exited during startup:\n{tail}")
        if _endpoint_healthy():
            return state, False
        time.sleep(1)
    clear_server()
    raise TimeoutError(f"{engine} server did not become ready within 10 minutes; see {log_path}")


def stream_completion(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    thinking: bool = False,
) -> Iterator[str]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token = delta.get("content") or delta.get("reasoning_content")
                if token:
                    yield token


def token_count(engine: str, model: str, text: str) -> int:
    if engine == "llama.cpp":
        body = {"content": text}
    elif engine == "vllm":
        body = {"model": model, "prompt": text}
    else:
        body = {"text": text}
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/tokenize",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        tokens = result.get("tokens", [])
        return len(tokens) if isinstance(tokens, list) else int(tokens)
    except Exception:
        return max(1, round(len(text.split()) * 1.3)) if text else 0
