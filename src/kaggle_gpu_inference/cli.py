from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil
from rich.console import Console
from rich.live import Live

from .config import HOME, ensure_dirs, read_state
from .engines import ENGINE_NAMES, clear_server, ensure_server, server_matches, stream_completion, token_count
from .models import context_estimates, estimated_model_size_bytes, load_gguf_config, load_hf_config, parse_model_ref, resolve_model
from .monitor import HardwareMonitor, aggregate_samples
from .reporting import append_csv, final_summary, live_dashboard


console = Console()


def setup_engine(engine: str) -> None:
    ensure_dirs()
    if engine == "llama.cpp":
        source = HOME / "vendor/llama.cpp"
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(source)],
                check=True,
            )
        subprocess.run(
            [
                "cmake", "-S", str(source), "-B", str(source / "build"),
                "-DGGML_CUDA=ON", "-DGGML_CUDA_NO_VMM=ON", "-DLLAMA_CURL=OFF",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            check=True,
        )
        subprocess.run(["cmake", "--build", str(source / "build"), "--target", "llama-server", "-j", "2"], check=True)
    else:
        package = "vllm" if engine == "vllm" else "sglang[all]"
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)


def command_run(args: argparse.Namespace) -> None:
    ref = parse_model_ref(args.model, args.filename)
    current = read_state()
    requested_signature = (ref.source, args.engine, args.gpus, args.context)
    current_signature = (
        current.get("source"), current.get("engine"), current.get("gpus"), current.get("context")
    )
    if current and requested_signature != current_signature:
        clear_server()
    model = resolve_model(ref, args.engine)
    reusable = server_matches(args.engine, model, ref.source, args.gpus, args.context)
    if current and requested_signature == current_signature and not reusable:
        clear_server()
    monitor = HardwareMonitor(args.gpus)
    initial = monitor.sample()
    model_bytes = estimated_model_size_bytes(ref, model)
    config = {**load_hf_config(ref), **{key: value for key, value in load_gguf_config(model).items() if value}}
    per_gpu_vram = int(initial.vram_total_gb * 1e9 / max(1, args.gpus))
    context_1, context_2 = context_estimates(model_bytes, config, per_gpu_vram)
    selected_context = context_1 if args.gpus == 1 else context_2
    available_ram = psutil.virtual_memory().available
    usable_vram = initial.vram_total_gb * 1e9 * 0.90 - initial.vram_used_gb * 1e9
    if not reusable and model_bytes and model_bytes > usable_vram:
        raise RuntimeError(
            f"Model weights ({model_bytes / 1e9:.2f} GB) exceed safe selected-GPU VRAM "
            f"({usable_vram / 1e9:.2f} GB). Use more GPUs or a smaller quantization."
        )
    if not reusable and model_bytes and model_bytes > available_ram * 0.80:
        raise RuntimeError(
            f"Model weights ({model_bytes / 1e9:.2f} GB) exceed the safe RAM loading limit "
            f"({available_ram * 0.80 / 1e9:.2f} GB)."
        )
    if selected_context and args.context > selected_context:
        raise RuntimeError(
            f"Requested context {args.context:,} exceeds the estimated safe maximum "
            f"{selected_context:,} for {args.gpus} GPU(s)."
        )
    state, reused = ensure_server(args.engine, model, ref.source, args.gpus, args.context)
    prompt_tokens = token_count(args.engine, model, args.prompt)
    started = time.perf_counter()
    first_token_at: float | None = None
    previous_token_at: float | None = None
    output_parts: list[str] = []
    intervals: list[float] = []
    samples = [initial]
    with Live(console=console, refresh_per_second=4, transient=False) as live:
        live.update(live_dashboard(ref.name, args.engine, args.gpus, context_1, context_2, "", initial, None, 0.0))
        for token in stream_completion(model, args.prompt, args.max_tokens, args.temperature):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            if previous_token_at is not None:
                intervals.append(now - previous_token_at)
            previous_token_at = now
            output_parts.append(token)
            sample = monitor.sample()
            samples.append(sample)
            speed = (1 / intervals[-1]) if intervals and intervals[-1] > 0 else 0.0
            live.update(live_dashboard(
                ref.name, args.engine, args.gpus, context_1, context_2,
                "".join(output_parts), sample, first_token_at - started, speed,
            ))
    finished = time.perf_counter()
    output = "".join(output_parts)
    final_sample = samples[-1]
    ttft = (first_token_at - started) if first_token_at else finished - started
    averages = aggregate_samples(samples)
    console.print(final_summary(ttft, intervals, final_sample, averages))
    output_tokens = token_count(args.engine, model, output)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": ref.name,
        "model_source": ref.source,
        "inference_engine": args.engine,
        "num_gpus": args.gpus,
        "server_reused": reused,
        "context_requested": args.context,
        "theoretical_max_context_1_gpu": context_1,
        "theoretical_max_context_2_gpu": context_2,
        "prompt_text": args.prompt,
        "prompt_length_words": len(args.prompt.split()),
        "prompt_length_tokens": prompt_tokens,
        "output": output,
        "output_length_words": len(output.split()),
        "output_length_tokens": output_tokens,
        "time_to_first_token_seconds": ttft,
        "time_to_first_token_ms": ttft * 1000,
        "generation_seconds": finished - (first_token_at or started),
        "token_speed_average": output_tokens / max(finished - (first_token_at or started), 1e-9),
        "token_speed_event_min": min((1 / value for value in intervals if value > 0), default=0),
        "token_speed_event_median": statistics.median((1 / value for value in intervals if value > 0)) if intervals else 0,
        "token_speed_event_max": max((1 / value for value in intervals if value > 0), default=0),
        **averages,
    }
    path = append_csv(record)
    console.print(f"Run logged to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kgpu", description="Kaggle multi-GPU LLM inference")
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser("setup", help="Install an inference engine")
    setup.add_argument("--engine", choices=ENGINE_NAMES, default="llama.cpp")
    run = subparsers.add_parser("run", help="Download/load a model and stream a response")
    run.add_argument("model")
    run.add_argument("--engine", choices=ENGINE_NAMES, default="llama.cpp")
    run.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    run.add_argument("--context", type=int, default=4096)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--filename", help="GGUF filename when model is an owner/repository ID")
    run.add_argument("--prompt", required=True)
    subparsers.add_parser("status", help="Show the persistent server state")
    subparsers.add_parser("clear-vram", help="Stop the model server and release VRAM")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "setup":
        setup_engine(args.engine)
    elif args.command == "run":
        command_run(args)
    elif args.command == "status":
        console.print_json(json.dumps(read_state()))
    elif args.command == "clear-vram":
        console.print("Stopped model server." if clear_server() else "No active model server.")


if __name__ == "__main__":
    main()