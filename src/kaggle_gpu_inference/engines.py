from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import psutil

from .config import HOME, PORT, RUNTIME_DIR, STATE_FILE, ensure_dirs, hf_token, read_state, write_state


ENGINE_NAMES = ("llama.cpp", "vllm", "sglang")
SPEC_TYPES = ("none", "draft-mtp")


@dataclass(frozen=True)
class StreamChunk:
    reasoning: str = ""
    content: str = ""
    output_tokens: int | None = None


def _server_command(
    engine: str,
    model: str,
    gpus: int,
    context: int,
    spec_type: str = "none",
    spec_draft_n_max: int = 2,
    accelerator_kind: str = "gpu",
) -> list[str]:
    if engine == "llama.cpp":
        if accelerator_kind != "gpu":
            raise RuntimeError("llama.cpp is supported only on GPU; use vllm or sglang on TPU")
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
        if spec_type != "none":
            command.extend([
                "--spec-type", spec_type,
                "--spec-draft-n-max", str(spec_draft_n_max),
            ])
        return command
    if engine == "vllm":
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
            "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", str(context),
            "--tensor-parallel-size", str(gpus), "--gpu-memory-utilization", "0.90",
        ]
    if engine == "sglang":
        if accelerator_kind == "tpu":
            return [
                sys.executable, "-u", "-m", "sgl_jax.launch_server", "--model-path", model,
                "--host", "127.0.0.1", "--port", str(PORT), "--context-length", str(context),
                "--tp-size", str(gpus), "--device", "tpu", "--dtype", "bfloat16",
                "--mem-fraction-static", "0.8",
            ]
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


def server_matches(
    engine: str,
    model: str,
    source: str,
    gpus: int,
    context: int,
    spec_type: str = "none",
    spec_draft_n_max: int = 2,
    accelerator_kind: str = "gpu",
) -> bool:
    expected = {
        "engine": engine,
        "model": model,
        "source": source,
        "gpus": gpus,
        "context": context,
        "spec_type": spec_type,
        "spec_draft_n_max": spec_draft_n_max,
        "accelerator": accelerator_kind,
    }
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


def ensure_server(
    engine: str,
    model: str,
    source: str,
    gpus: int,
    context: int,
    spec_type: str = "none",
    spec_draft_n_max: int = 2,
    accelerator_kind: str = "gpu",
) -> tuple[dict, bool]:
    ensure_dirs()
    signature = {
        "engine": engine,
        "model": model,
        "source": source,
        "gpus": gpus,
        "context": context,
        "spec_type": spec_type,
        "spec_draft_n_max": spec_draft_n_max,
        "accelerator": accelerator_kind,
    }
    state = read_state()
    if (
        _state_process_running(state)
        and all(state.get(key) == value for key, value in signature.items())
        and _endpoint_healthy()
    ):
        return state, True
    clear_server()
    command = _server_command(
        engine, model, gpus, context, spec_type, spec_draft_n_max, accelerator_kind
    )
    log_path = RUNTIME_DIR / "server.log"
    environment = os.environ.copy()
    if accelerator_kind == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(gpus))
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.setdefault("JAX_COMPILATION_CACHE_DIR", str(RUNTIME_DIR / "jax_cache"))
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
) -> Iterator[StreamChunk]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
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
            usage = event.get("usage") or {}
            output_tokens = usage.get("completion_tokens")
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                reasoning = delta.get("reasoning_content") or ""
                content = delta.get("content") or ""
                if reasoning or content:
                    yield StreamChunk(
                        reasoning=reasoning,
                        content=content,
                        output_tokens=int(output_tokens) if output_tokens is not None else None,
                    )
            elif output_tokens is not None:
                yield StreamChunk(output_tokens=int(output_tokens))


def token_count(engine: str, model: str, text: str, chat: bool = False) -> int:
    if engine == "llama.cpp":
        body = {"content": text}
    elif engine == "vllm":
        body = {"model": model}
        body["messages" if chat else "prompt"] = (
            [{"role": "user", "content": text}] if chat else text
        )
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
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model)
            if chat and tokenizer.chat_template:
                tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            else:
                tokens = tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)
        except Exception:
            return max(1, round(len(text.split()) * 1.3)) if text else 0
