"""AkinVox native English/German multispeaker inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("akinvox-kokoro-multispeaker")
except PackageNotFoundError:
    __version__ = "1.0.0"


def __getattr__(name):
    if name in {"KokoroMultispeaker", "SpeechChunk"}:
        from . import shared

        return getattr(shared, name)
    raise AttributeError(name)
