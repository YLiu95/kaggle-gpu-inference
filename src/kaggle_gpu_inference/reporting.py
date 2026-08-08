from __future__ import annotations

import csv
import shutil
import statistics
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import LOG_DIR, ensure_dirs
from .monitor import Sample


KAGGLE_WORKING = Path("/kaggle/working")


def _metric(label: str, value: str) -> Text:
    text = Text()
    text.append(f"{label}\n", style="dim")
    text.append(value, style="bold cyan")
    return text


def live_dashboard(
    model_name: str,
    engine: str,
    accelerator_name: str,
    device_count: int,
    max_context: int,
    input_tokens: int,
    output_tokens: int,
    max_output_tokens: int,
    reasoning: str,
    output: str,
    show_reasoning: bool,
    sample: Sample,
    ttft: float | None,
    token_speed: float,
) -> Group:
    top = Columns([
        _metric("Model", model_name),
        _metric("Engine", engine),
        _metric("Accelerator", f"{accelerator_name} x {device_count}"),
        _metric("Max context window", f"{max_context:,}"),
        _metric("Input tokens", f"{input_tokens:,}"),
        _metric("Output tokens", f"{output_tokens:,}"),
        _metric("Max output tokens", f"{max_output_tokens:,}"),
        _metric("Time to first token", f"{ttft:.3f} s ({ttft * 1000:.0f} ms)" if ttft else "waiting"),
        _metric("Token speed", f"{token_speed:.2f} token/s"),
        _metric("Compute throughput*", f"{sample.estimated_flops / 1e12:.2f} TFLOPS"),
        _metric("GPU memory throughput*", f"{sample.estimated_gpu_memory_gbs:.2f} GB/s"),
        _metric("GPU-CPU throughput", f"{sample.gpu_cpu_gbs:.3f} GB/s"),
        _metric("Accelerator memory", f"{sample.vram_used_gb:.2f}/{sample.vram_total_gb:.2f} GB ({sample.vram_percent:.1f}%)"),
        _metric("CPU", f"{sample.cpu_percent:.1f}% @ {sample.cpu_speed_ghz:.2f} GHz"),
        _metric("CPU RAM", f"{sample.ram_used_gb:.2f}/{sample.ram_total_gb:.2f} GB ({sample.ram_percent:.1f}%)"),
    ], equal=True, expand=True)
    panels = []
    if show_reasoning:
        reasoning_stream = Text(reasoning or "Waiting for reasoning tokens...", overflow="fold")
        panels.append(Panel(reasoning_stream, title="Reasoning tokens", border_style="yellow"))
    response_stream = Text(output or "Waiting for response tokens...", overflow="fold")
    response_title = "Response tokens" if show_reasoning else "Streaming tokens"
    panels.append(Panel(response_stream, title=response_title, border_style="green"))
    panels.append(Panel(top, title="Inference", border_style="cyan"))
    return Group(*panels)


def final_summary(ttft: float, token_intervals: list[float], sample: Sample, averages: dict[str, float]) -> Panel:
    speeds = [1 / interval for interval in token_intervals if interval > 0]
    table = Table(show_header=False, box=None, expand=True)
    table.add_column(style="dim")
    table.add_column(style="bold")
    table.add_row("Time to first token", f"{ttft:.3f} s ({ttft * 1000:.0f} ms)")
    if speeds:
        table.add_row(
            "Token speed min / median / max / average",
            f"{min(speeds):.2f} / {statistics.median(speeds):.2f} / {max(speeds):.2f} / {statistics.fmean(speeds):.2f} token/s",
        )
    table.add_row("Average compute throughput*", f"{averages.get('estimated_flops_avg', 0) / 1e12:.2f} TFLOPS")
    table.add_row("Average GPU memory throughput*", f"{averages.get('estimated_gpu_memory_gbs_avg', 0):.2f} GB/s")
    table.add_row("Average GPU-CPU throughput", f"{averages.get('gpu_cpu_gbs_avg', 0):.3f} GB/s")
    table.add_row("Final VRAM", f"{sample.vram_used_gb:.2f}/{sample.vram_total_gb:.2f} GB ({sample.vram_percent:.1f}%)")
    table.add_row(
        "CPU average / RAM average",
        f"{averages.get('cpu_percent_avg', 0):.1f}% / {averages.get('ram_used_gb_avg', 0):.2f} GB ({averages.get('ram_percent_avg', 0):.1f}%)",
    )
    table.add_row("*", "Estimated from utilization x Tesla T4 peak specification")
    return Panel(table, title="Completed", border_style="blue")


def append_csv(record: dict[str, Any]) -> Path:
    ensure_dirs()
    path = LOG_DIR / "inference_runs.csv"
    existing_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
    fieldnames.extend(key for key in record if key not in fieldnames)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerow(record)
    temporary.replace(path)
    if KAGGLE_WORKING.is_dir():
        mirror = KAGGLE_WORKING / path.name
        mirror_temporary = mirror.with_suffix(".tmp")
        shutil.copy2(path, mirror_temporary)
        mirror_temporary.replace(mirror)
    return path
