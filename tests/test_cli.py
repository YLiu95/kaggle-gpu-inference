import pytest

from kaggle_gpu_inference.cli import build_parser, normalize_cli_args, parse_bool


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), ("1", True), ("false", False), ("off", False)],
)
def test_parse_bool(value: str, expected: bool) -> None:
    assert parse_bool(value) is expected


def test_thinking_cli_forms() -> None:
    parser = build_parser()
    base = ["run", "owner/model", "--prompt", "test"]
    assert parser.parse_args([*base, "--thinking"]).thinking is True
    assert parser.parse_args([*base, "--thinking", "true"]).thinking is True
    assert parser.parse_args([*base, "--thinking", "false"]).thinking is False
    assert parser.parse_args([*base, "--no-thinking"]).thinking is False


def test_bare_thinking_before_model_is_not_consumed() -> None:
    arguments = ["run", "--thinking", "owner/model", "--prompt", "test"]
    parsed = build_parser().parse_args(normalize_cli_args(arguments))
    assert parsed.model == "owner/model"
    assert parsed.thinking is True


def test_parse_bool_rejects_unknown_value() -> None:
    with pytest.raises(Exception):
        parse_bool("maybe")