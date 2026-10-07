"""Invalid CLI input must fail before downloads or model allocation."""

import sys
import pytest
from kokoro_multispeaker import cli, shared


@pytest.fixture
def no_model(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid input reached model construction")

    monkeypatch.setattr(shared, "KokoroMultispeaker", unexpected)


@pytest.mark.parametrize("text", ["   ", "???"])
def test_empty_or_unspoken_input_is_rejected(no_model, monkeypatch, text):
    monkeypatch.setattr(sys, "argv", ["kokoro-multispeaker", "--text", text])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_input_file_cannot_be_overwritten(no_model, monkeypatch, tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("Keep this source text.")
    monkeypatch.setattr(
        sys,
        "argv",
        ["kokoro-multispeaker", "--text-file", str(path), "--output", str(path)],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert path.read_text() == "Keep this source text."


def test_missing_text_file_is_rejected_before_loading(no_model, monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["kokoro-multispeaker", "--text-file", str(tmp_path / "missing.txt")],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
