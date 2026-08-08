import csv

from kaggle_gpu_inference import reporting


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
