"""Downloaded language selection must agree with the pinned release catalog."""

import json

import huggingface_hub
import pytest

from kokoro_multispeaker import bundles


@pytest.fixture
def snapshot(monkeypatch, tmp_path):
    registry = {}
    for language in ("en", "de"):
        folder = tmp_path / "adapters" / language
        folder.mkdir(parents=True)
        (folder / "adapter.pt").write_bytes(language.encode())
        (folder / "config.json").write_text("{}")
        registry[language] = {
            "repo": "AKinvox/kokoro-multispeaker",
            "revision": "pinned-revision",
            "path": "adapters/" + language,
            "adapter_sha256": bundles.sha(folder / "adapter.pt"),
            "config_sha256": bundles.sha(folder / "config.json"),
            "speaker_names": ["AV_example"],
        }
    (tmp_path / "releases.json").write_text(json.dumps(registry))
    catalog = {
        "format": "akinvox-multispeaker-catalog/v1",
        "base_sha256": bundles.BASE_SHA256,
        "languages": {
            language: {
                key: value
                for key, value in item.items()
                if key not in {"repo", "revision"}
            }
            for language, item in registry.items()
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(catalog))
    calls = []

    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(bundles, "__file__", str(tmp_path / "bundles.py"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return tmp_path, catalog, calls


@pytest.mark.parametrize("language", ["en", "de"])
def test_catalog_is_retrieved_with_only_the_selected_adapter(snapshot, language):
    root, _, calls = snapshot
    assert bundles._default_bundle(language) == root / "adapters" / language
    repo, kwargs = calls[0]
    assert repo == "AKinvox/kokoro-multispeaker"
    assert kwargs["revision"] == "pinned-revision"
    assert set(kwargs["allow_patterns"]) == {
        "config.json",
        "adapters/" + language + "/adapter.pt",
        "adapters/" + language + "/config.json",
        "adapters/" + language + "/SHA256SUMS",
    }


@pytest.mark.parametrize("field", ["base_sha256", "path", "speaker_names"])
def test_catalog_disagreement_fails_before_adapter_use(snapshot, field):
    root, catalog, _ = snapshot
    if field == "base_sha256":
        catalog[field] = "wrong-base"
    else:
        catalog["languages"]["en"][field] = "wrong-entry"
    (root / "config.json").write_text(json.dumps(catalog))
    (root / "adapters/en/adapter.pt").unlink()
    with pytest.raises(ValueError, match="catalog differs"):
        bundles._default_bundle("en")


def test_missing_catalog_is_not_silently_ignored(snapshot):
    root, _, _ = snapshot
    (root / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        bundles._default_bundle("en")
