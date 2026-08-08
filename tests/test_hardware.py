from kaggle_gpu_inference import hardware


def test_tpu_detection_precedes_nvml(monkeypatch) -> None:
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5e-8")
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0,1,2,3,4,5,6,7")

    profile = hardware.detect_accelerator()

    assert profile.kind == "tpu"
    assert profile.name == "v5e-8"
    assert profile.device_count == 8
    assert profile.memory_per_device_bytes == 16 * 1024**3


def test_force_tpu_uses_kaggle_v5e_defaults(monkeypatch) -> None:
    for name in ("TPU_NAME", "TPU_ACCELERATOR_TYPE", "TPU_WORKER_ID", "TPU_VISIBLE_CHIPS"):
        monkeypatch.delenv(name, raising=False)

    profile = hardware.detect_accelerator(force_tpu=True)

    assert profile.kind == "tpu"
    assert profile.name == "v5e-8"
    assert profile.device_count == 8
    assert profile.total_memory_bytes == 128 * 1024**3


def test_tpu_topology_bounds_determine_device_count(monkeypatch) -> None:
    monkeypatch.setenv("TPU_ACCELERATOR_TYPE", "v5e")
    monkeypatch.setenv("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,2")

    assert hardware.detect_accelerator().device_count == 8