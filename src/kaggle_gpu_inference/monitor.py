from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import psutil
from pynvml import (
    NVML_PCIE_UTIL_RX_BYTES,
    NVML_PCIE_UTIL_TX_BYTES,
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetPcieThroughput,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
)


T4_FP16_TENSOR_FLOPS = 65e12
T4_MEMORY_BANDWIDTH_GBS = 320.0


@dataclass
class Sample:
    timestamp: float
    gpu_util_percent: float
    gpu_memory_util_percent: float
    vram_used_gb: float
    vram_total_gb: float
    vram_percent: float
    estimated_flops: float
    estimated_gpu_memory_gbs: float
    gpu_cpu_gbs: float
    cpu_percent: float
    cpu_speed_ghz: float
    ram_used_gb: float
    ram_total_gb: float
    ram_percent: float
    gpu_0_vram_gb: float = 0.0
    gpu_0_util_percent: float = 0.0
    gpu_1_vram_gb: float = 0.0
    gpu_1_util_percent: float = 0.0


class HardwareMonitor:
    def __init__(self, gpu_count: int) -> None:
        nvmlInit()
        self.handles = [nvmlDeviceGetHandleByIndex(index) for index in range(min(gpu_count, nvmlDeviceGetCount()))]
        psutil.cpu_percent(interval=None)

    def sample(self) -> Sample:
        gpu_utils: list[float] = []
        memory_utils: list[float] = []
        used: list[int] = []
        total: list[int] = []
        pcie_kbs = 0
        for handle in self.handles:
            utilization = nvmlDeviceGetUtilizationRates(handle)
            memory = nvmlDeviceGetMemoryInfo(handle)
            gpu_utils.append(float(utilization.gpu))
            memory_utils.append(float(utilization.memory))
            used.append(memory.used)
            total.append(memory.total)
            try:
                pcie_kbs += nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES)
                pcie_kbs += nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES)
            except NVMLError:
                pass
        virtual_memory = psutil.virtual_memory()
        frequency = psutil.cpu_freq()
        used_total = sum(used)
        vram_total = sum(total)
        mean_gpu = statistics.fmean(gpu_utils) if gpu_utils else 0.0
        mean_memory = statistics.fmean(memory_utils) if memory_utils else 0.0
        return Sample(
            timestamp=time.time(),
            gpu_util_percent=mean_gpu,
            gpu_memory_util_percent=mean_memory,
            vram_used_gb=used_total / 1e9,
            vram_total_gb=vram_total / 1e9,
            vram_percent=(100 * used_total / vram_total) if vram_total else 0.0,
            estimated_flops=sum(value / 100 * T4_FP16_TENSOR_FLOPS for value in gpu_utils),
            estimated_gpu_memory_gbs=sum(value / 100 * T4_MEMORY_BANDWIDTH_GBS for value in memory_utils),
            gpu_cpu_gbs=pcie_kbs / 1e6,
            cpu_percent=psutil.cpu_percent(interval=None),
            cpu_speed_ghz=(frequency.current / 1000) if frequency else 0.0,
            ram_used_gb=virtual_memory.used / 1e9,
            ram_total_gb=virtual_memory.total / 1e9,
            ram_percent=virtual_memory.percent,
            gpu_0_vram_gb=(used[0] / 1e9) if used else 0.0,
            gpu_0_util_percent=gpu_utils[0] if gpu_utils else 0.0,
            gpu_1_vram_gb=(used[1] / 1e9) if len(used) > 1 else 0.0,
            gpu_1_util_percent=gpu_utils[1] if len(gpu_utils) > 1 else 0.0,
        )


def aggregate_samples(samples: list[Sample]) -> dict[str, float]:
    if not samples:
        return {}
    fields = asdict(samples[0]).keys()
    result: dict[str, float] = {}
    for field in fields:
        if field == "timestamp":
            continue
        values = [float(getattr(sample, field)) for sample in samples]
        result[f"{field}_avg"] = statistics.fmean(values)
        result[f"{field}_min"] = min(values)
        result[f"{field}_max"] = max(values)
    return result
