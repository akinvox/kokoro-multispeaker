"""Download and validate the stock model and language adapter bundles."""

import hashlib
import json
import re
from pathlib import Path

import torch

BASE_REPO = "hexgrad/Kokoro-82M"
BASE_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
BASE_SHA256 = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"
CONFIG_SHA256 = "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f"
HEART_SHA256 = "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_bundle(path, language):
    root = Path(path)
    checked = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        if (
            name not in {"adapter.pt", "config.json"}
            or name in checked
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            raise ValueError("Invalid adapter checksum manifest")
        if sha(root / name) != digest:
            raise ValueError("Checksum mismatch: " + name)
        checked.add(name)
    if checked != {"adapter.pt", "config.json"}:
        raise ValueError("Incomplete adapter checksum manifest")
    config = json.loads((root / "config.json").read_text())
    if (
        config.get("format") != "akinvox-stock-language-bundle/v1"
        or config.get("language") != language
        or config.get("base_sha256") != BASE_SHA256
    ):
        raise ValueError("Incompatible language bundle")
    release = torch.load(
        root / "adapter.pt", map_location="cpu", weights_only=True, mmap=True
    )
    names, bank = release.get("speaker_names"), release.get("voicebank")
    if (
        release.get("contract") != "akinvox-stock-multispeaker-adapter/v1"
        or release.get("task") != language + "_multispeaker"
        or release.get("base_sha256") != BASE_SHA256
        or not isinstance(names, list)
        or not names
        or len(names) > 1000
        or any(
            not isinstance(n, str) or not re.fullmatch("AV_[a-z_]+", n) for n in names
        )
        or len(set(names)) != len(names)
        or names != config.get("speaker_names")
        or not isinstance(bank, torch.Tensor)
        or bank.shape != (len(names), 510, 256)
        or bank.dtype != torch.float32
        or not torch.isfinite(bank).all()
        or not torch.equal(bank, bank[:, 0:1].expand_as(bank))
        or len(release.get("targets", [])) != 201
        or len(release.get("state", {})) != 402
    ):
        raise ValueError("Invalid fixed voicebank or adapter inventory")
    if set(release) != {
        "contract",
        "task",
        "base_sha256",
        "update",
        "targets",
        "state",
        "voicebank",
        "speaker_names",
        "source_checkpoint_sha256",
        "voice_conditioning",
    }:
        raise ValueError("Unexpected inference adapter fields")
    return release, config


def _default_bundle(language):
    registry = json.loads(Path(__file__).with_name("releases.json").read_text())
    if language not in registry:
        raise ValueError(
            "No published adapter for " + language + "; supply adapter_dirs"
        )
    item = registry[language]
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            item["repo"],
            revision=item["revision"],
            allow_patterns=["config.json"]
            + [
                item["path"] + "/" + x
                for x in ["adapter.pt", "config.json", "SHA256SUMS"]
            ],
        )
    )
    catalog = json.loads((snapshot / "config.json").read_text())
    entry = catalog.get("languages", {}).get(language)
    expected = {
        key: item[key]
        for key in ("path", "adapter_sha256", "config_sha256", "speaker_names")
    }
    if (
        catalog.get("format") != "akinvox-multispeaker-catalog/v1"
        or catalog.get("base_sha256") != BASE_SHA256
        or entry != expected
    ):
        raise ValueError("Published language catalog differs from its release pin")
    root = snapshot / entry["path"]
    if (
        sha(root / "adapter.pt") != entry["adapter_sha256"]
        or sha(root / "config.json") != entry["config_sha256"]
    ):
        raise ValueError("Published bundle differs from its immutable release pin")
    return root
