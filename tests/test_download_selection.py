"""A single-language request must not fetch an unrelated adapter."""

import pytest
from kokoro_multispeaker import shared


class StopBeforeModelLoad(Exception):
    pass


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    fetched = []

    def download(language):
        fetched.append(language)
        return tmp_path / language

    def stop(*args):
        raise StopBeforeModelLoad

    monkeypatch.setattr(shared, "_default_bundle", download)
    monkeypatch.setattr(shared, "_checked_bundle", stop)
    return fetched


def test_one_language_does_not_download_the_other(downloads):
    with pytest.raises(StopBeforeModelLoad):
        shared.KokoroMultispeaker(languages=["en"])
    assert downloads == ["en"]


def test_default_keeps_both_available_languages(downloads):
    with pytest.raises(StopBeforeModelLoad):
        shared.KokoroMultispeaker()
    assert set(downloads) == {"en", "de"}


@pytest.mark.parametrize("languages", [[], ["en", "en"], ["unknown"], "en"])
def test_invalid_selection_fails_before_any_download(downloads, languages):
    with pytest.raises(ValueError, match="unique published languages"):
        shared.KokoroMultispeaker(languages=languages)
    assert downloads == []


def test_local_files_and_download_selection_are_not_mixed(downloads, tmp_path):
    with pytest.raises(ValueError, match="adapter_dirs"):
        shared.KokoroMultispeaker({"en": tmp_path}, languages=["en"])
    assert downloads == []
