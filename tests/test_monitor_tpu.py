from kaggle_gpu_inference.hardware import Accelerator
from kaggle_gpu_inference.monitor import HardwareMonitor


def test_tpu_monitor_does_not_initialize_nvml() -> None:
    monitor = HardwareMonitor(Accelerator("tpu", "v5e-8", 8, 16 * 1024**3))
    sample = monitor.sample()

    assert monitor.nvml is None
    assert sample.vram_total_gb > 137
    assert sample.gpu_util_percent == 0