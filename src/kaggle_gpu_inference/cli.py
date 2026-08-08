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
from .engines import ENGINE_NAMES, SPEC_TYPES, clear_server, ensure_server, server_matches, stream_completion, token_count
from .models import context_estimates, estimated_model_size_bytes, load_gguf_config, load_hf_config, parse_model_ref, resolve_model
from .monitor import HardwareMonitor, aggregate_samples
from .reporting import append_csv, final_summary, live_dashboard


console = Console()
BOOLEAN_VALUES = {"true", "false", "1", "0", "yes", "no", "on", "off"}
MTP_GPU_HEADROOM_BYTES = 500_000_000
MTP_RAM_HEADROOM_BYTES = 2_000_000_000
MTP_KV_HEADROOM_PER_DRAFT_TOKEN_BYTES = 64_000_000


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def normalize_cli_args(arguments: list[str]) -> list[str]:
    normalized: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "--thinking":
            next_argument = arguments[index + 1].lower() if index + 1 < len(arguments) else ""
            if next_argument not in BOOLEAN_VALUES:
                normalized.append("--thinking=true")
                continue
        normalized.append(argument)
    return normalized


def spec_draft_count(value: str) -> int:
    count = int(value)
    if not 1 <= count <= 6:
        raise argparse.ArgumentTypeError("expected a value from 1 through 6")
    return count


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
    if args.engine != "llama.cpp" and args.spec_type != "none":
        raise ValueError("--spec-type is currently supported only by the llama.cpp engine")
    ref = parse_model_ref(args.model, args.filename)
    current = read_state()
    requested_signature = (
        ref.source, args.engine, args.gpus, args.context, args.spec_type, args.spec_draft_n_max
    )
    current_signature = (
        current.get("source"), current.get("engine"), current.get("gpus"), current.get("context"),
        current.get("spec_type", "none"), current.get("spec_draft_n_max", 2),
    )
    if current and requested_signature != current_signature:
        clear_server()
    model = resolve_model(ref, args.engine)
    reusable = server_matches(
        args.engine, model, ref.source, args.gpus, args.context,
        args.spec_type, args.spec_draft_n_max,
    )
    if current and requested_signature == current_signature and not reusable:
        clear_server()
    monitor = HardwareMonitor(args.gpus)
    initial = monitor.sample()
    model_bytes = estimated_model_size_bytes(ref, model)
    config = {**load_hf_config(ref), **{key: value for key, value in load_gguf_config(model).items() if value}}
    per_gpu_vram = int(initial.vram_total_gb * 1e9 / max(1, args.gpus))
    mtp_gpu_headroom = MTP_GPU_HEADROOM_BYTES if args.spec_type == "draft-mtp" else 0
    mtp_ram_headroom = MTP_RAM_HEADROOM_BYTES if args.spec_type == "draft-mtp" else 0
    mtp_kv_headroom = (
        MTP_KV_HEADROOM_PER_DRAFT_TOKEN_BYTES * args.spec_draft_n_max
        if args.spec_type == "draft-mtp" else 0
    )
    gpu_capacity_bytes = model_bytes + mtp_gpu_headroom
    ram_capacity_bytes = model_bytes + mtp_ram_headroom
    context_1, context_2 = context_estimates(model_bytes + mtp_kv_headroom, config, per_gpu_vram)
    selected_context = context_1 if args.gpus == 1 else context_2
    available_ram = psutil.virtual_memory().available
    usable_vram = initial.vram_total_gb * 1e9 * 0.90 - initial.vram_used_gb * 1e9
    if not reusable and model_bytes and gpu_capacity_bytes > usable_vram:
        raise RuntimeError(
            f"Model weights plus GPU workspace ({gpu_capacity_bytes / 1e9:.2f} GB) "
            f"exceed safe selected-GPU VRAM "
            f"({usable_vram / 1e9:.2f} GB). Use more GPUs or a smaller quantization."
        )
    if not reusable and model_bytes and ram_capacity_bytes > available_ram * 0.80:
        raise RuntimeError(
            f"Model weights plus runtime headroom ({ram_capacity_bytes / 1e9:.2f} GB) "
            f"exceed the safe RAM loading limit "
            f"({available_ram * 0.80 / 1e9:.2f} GB)."
        )
    if selected_context and args.context > selected_context:
        raise RuntimeError(
            f"Requested context {args.context:,} exceeds the estimated safe maximum "
            f"{selected_context:,} for {args.gpus} GPU(s)."
        )
    state, reused = ensure_server(
        args.engine, model, ref.source, args.gpus, args.context,
        args.spec_type, args.spec_draft_n_max,
    )
    prompt_tokens = token_count(args.engine, model, args.prompt)
    started = time.perf_counter()
    first_token_at: float | None = None
    previous_token_at: float | None = None
    reasoning_parts: list[str] = []
    output_parts: list[str] = []
    intervals: list[float] = []
    samples = [initial]
    with Live(
        console=console,
        refresh_per_second=4,
        transient=False,
        vertical_overflow="visible",
    ) as live:
        live.update(live_dashboard(
            ref.name, args.engine, args.gpus, context_1, context_2,
            "", "", args.thinking, initial, None, 0.0,
        ))
        for chunk in stream_completion(
            model, args.prompt, args.max_tokens, args.temperature, thinking=args.thinking
        ):
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            if previous_token_at is not None:
                intervals.append(now - previous_token_at)
            previous_token_at = now
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            if chunk.content:
                output_parts.append(chunk.content)
            sample = monitor.sample()
            samples.append(sample)
            speed = (1 / intervals[-1]) if intervals and intervals[-1] > 0 else 0.0
            live.update(live_dashboard(
                ref.name, args.engine, args.gpus, context_1, context_2,
                "".join(reasoning_parts), "".join(output_parts), args.thinking,
                sample, first_token_at - started, speed,
            ))
    finished = time.perf_counter()
    reasoning = "".join(reasoning_parts)
    output = "".join(output_parts)
    final_sample = samples[-1]
    ttft = (first_token_at - started) if first_token_at else finished - started
    averages = aggregate_samples(samples)
    console.print(final_summary(ttft, intervals, final_sample, averages))
    reasoning_tokens = token_count(args.engine, model, reasoning)
    response_tokens = token_count(args.engine, model, output)
    output_tokens = reasoning_tokens + response_tokens
    output_words = len(reasoning.split()) + len(output.split())
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": ref.name,
        "model_source": ref.source,
        "inference_engine": args.engine,
        "num_gpus": args.gpus,
        "spec_type": args.spec_type,
        "spec_draft_n_max": args.spec_draft_n_max,
        "mtp_enabled": args.spec_type == "draft-mtp",
        "server_reused": reused,
        "context_requested": args.context,
        "theoretical_max_context_1_gpu": context_1,
        "theoretical_max_context_2_gpu": context_2,
        "prompt_text": args.prompt,
        "prompt_length_words": len(args.prompt.split()),
        "prompt_length_tokens": prompt_tokens,
        "thinking_enabled": args.thinking,
        "reasoning": reasoning,
        "reasoning_length_words": len(reasoning.split()),
        "reasoning_length_tokens": reasoning_tokens,
        "output": output,
        "response_length_words": len(output.split()),
        "response_length_tokens": response_tokens,
        "generated_text": reasoning + output,
        "output_length_words": output_words,
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
    run.add_argument(
        "--spec-type",
        choices=SPEC_TYPES,
        default="none",
        help="Speculative decoding mode; use draft-mtp for an MTP model",
    )
    run.add_argument(
        "--spec-draft-n-max",
        type=spec_draft_count,
        default=2,
        metavar="N",
        help="Maximum MTP draft tokens, 1-6 (default: 2)",
    )
    run.add_argument(
        "--thinking",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        metavar="{true,false}",
        help="Enable reasoning mode; accepts true/false (default: false)",
    )
    run.add_argument("--no-thinking", dest="thinking", action="store_false", help=argparse.SUPPRESS)
    run.add_argument("--filename", help="GGUF filename when model is an owner/repository ID")
    run.add_argument("--prompt", required=True)
    subparsers.add_parser("status", help="Show the persistent server state")
    subparsers.add_parser("clear-vram", help="Stop the model server and release VRAM")
    return parser


def main() -> None:
    args = build_parser().parse_args(normalize_cli_args(sys.argv[1:]))
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