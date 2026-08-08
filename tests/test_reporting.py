import csv

from rich.console import Console

from kaggle_gpu_inference import reporting
from kaggle_gpu_inference.monitor import Sample


def test_stream_panel_preserves_long_output() -> None:
    sample = Sample(
        timestamp=0,
        gpu_util_percent=0,
        gpu_memory_util_percent=0,
        vram_used_gb=1,
        vram_total_gb=16,
        vram_percent=6.25,
        estimated_flops=0,
        estimated_gpu_memory_gbs=0,
        gpu_cpu_gbs=0,
        cpu_percent=0,
        cpu_speed_ghz=2,
        ram_used_gb=2,
        ram_total_gb=32,
        ram_percent=6.25,
    )
    output = "\n".join(f"generated-line-{index:02d}" for index in range(30))
    console = Console(record=True, width=100, height=60)

    console.print(reporting.live_dashboard(
        "model", "llama.cpp", "Tesla T4", 1, 8192, 100, 30, 8092,
        "reasoning", output, True, sample, 0.1, 20
    ))

    rendered = console.export_text()
    assert "Reasoning tokens" in rendered
    assert "Response tokens" in rendered
    assert "reasoning" in rendered
    assert rendered.index("Reasoning tokens") < rendered.index("Response tokens") < rendered.index("Inference")
    assert "generated-line-00" in rendered
    assert "generated-line-29" in rendered
    assert "Max context window" in rendered
    assert "Input tokens" in rendered
    assert "Max output tokens" in rendered


def test_append_csv_preserves_old_rows_when_schema_grows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(reporting, "LOG_DIR", tmp_path)
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.setattr(reporting, "KAGGLE_WORKING", working)
    path = reporting.append_csv({"model": "first"})
    reporting.append_csv({"model": "second", "new_metric": 42})

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"model": "first", "new_metric": ""},
        {"model": "second", "new_metric": "42"},
    ]
    assert (working / path.name).read_bytes() == path.read_bytes()
    assert not (working / "inference_runs.tmp").exists()
