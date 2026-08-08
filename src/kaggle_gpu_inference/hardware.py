from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


TPU_MEMORY_BYTES = {
    "v5e": 16 * 1024**3,
    "v4": 32 * 1024**3,
    "v6e": 32 * 1024**3,
}


@dataclass(frozen=True)
class Accelerator:
    kind: str
    name: str
    device_count: int
    memory_per_device_bytes: int

    @property
    def total_memory_bytes(self) -> int:
        return self.device_count * self.memory_per_device_bytes


def _tpu_requested(environment: dict[str, str]) -> bool:
    return any(
        environment.get(name, "").lower() == "tpu" or environment.get(name)
        for name in (
            "TPU_NAME",
            "TPU_ACCELERATOR_TYPE",
            "TPU_WORKER_ID",
            "TPU_VISIBLE_CHIPS",
            "TPU_CHIPS_PER_HOST_BOUNDS",
        )
    ) or environment.get("PJRT_DEVICE", "").lower() == "tpu" or any(Path("/dev").glob("accel*"))


def _tpu_profile(environment: dict[str, str]) -> Accelerator:
    accelerator_type = environment.get("TPU_ACCELERATOR_TYPE", "v5e-8").lower()
    family = next((name for name in TPU_MEMORY_BYTES if name in accelerator_type), "v5e")
    visible = environment.get("TPU_VISIBLE_CHIPS", "")
    if visible:
        device_count = len([chip for chip in visible.split(",") if chip.strip()])
    elif bounds := environment.get("TPU_CHIPS_PER_HOST_BOUNDS", ""):
        dimensions = [int(value) for value in bounds.split(",") if value.isdigit()]
        device_count = 1
        for dimension in dimensions:
            device_count *= dimension
    else:
        suffix = accelerator_type.rsplit("-", 1)[-1]
        device_count = int(suffix) if suffix.isdigit() else 8
    return Accelerator("tpu", accelerator_type, device_count, TPU_MEMORY_BYTES[family])


def detect_accelerator(force_tpu: bool = False) -> Accelerator:
    environment = dict(os.environ)
    if force_tpu or _tpu_requested(environment):
        return _tpu_profile(environment)

    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlInit,
        )

        nvmlInit()
        device_count = nvmlDeviceGetCount()
        if device_count:
            handle = nvmlDeviceGetHandleByIndex(0)
            name = nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            return Accelerator(
                "gpu", str(name), device_count, int(nvmlDeviceGetMemoryInfo(handle).total)
            )
    except Exception:
        pass
    return Accelerator("cpu", "CPU", 0, 0)