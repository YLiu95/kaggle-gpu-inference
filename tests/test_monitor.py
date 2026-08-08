from kaggle_gpu_inference.monitor import Sample, aggregate_samples


def make_sample(vram: float, cpu: float) -> Sample:
    return Sample(
        timestamp=1,
        gpu_util_percent=50,
        gpu_memory_util_percent=25,
        vram_used_gb=vram,
        vram_total_gb=32,
        vram_percent=vram / 32 * 100,
        estimated_flops=32.5e12,
        estimated_gpu_memory_gbs=80,
        gpu_cpu_gbs=1,
        cpu_percent=cpu,
        cpu_speed_ghz=2,
        ram_used_gb=4,
        ram_total_gb=32,
        ram_percent=12.5,
    )


def test_aggregate_samples() -> None:
    result = aggregate_samples([make_sample(10, 20), make_sample(12, 40)])
    assert result["vram_used_gb_avg"] == 11
    assert result["vram_used_gb_min"] == 10
    assert result["vram_used_gb_max"] == 12
    assert result["cpu_percent_avg"] == 30


def test_aggregate_empty_samples() -> None:
    assert aggregate_samples([]) == {}
