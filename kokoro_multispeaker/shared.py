"""Python API for speech, streaming and language adapter selection."""

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from kokoro import KModel
from . import native_runtime as native
from .frontend import GermanFrontend, EnglishFrontend, spoken_text, model_tokens
from .context import MAX_CHUNK_SAMPLES, prepare_context_text, render_context
from .language_weights import ROOTS, SharedAdapters, share_stock_graph

from .bundles import (
    BASE_REPO,
    BASE_REVISION,
    BASE_SHA256,
    CONFIG_SHA256,
    HEART_SHA256,
    sha,
    _checked_bundle,
    _default_bundle,
)


class SharedStockKModel(KModel):
    adapters = None

    def forward_with_tokens(self, *args, **kwargs):
        if self.adapters is None:
            return super().forward_with_tokens(*args, **kwargs)
        with self.adapters.selection():
            return super().forward_with_tokens(*args, **kwargs)


@dataclass(frozen=True)
class SpeechChunk:
    """One audio chunk; timings are relative to this chunk's audio."""

    audio: np.ndarray
    timings: list | None
    context_used: bool


def split_phones(phones):
    """Split phonemes at spaces when source-word timings are unavailable."""
    chunks = []
    while len(phones) > 508:
        boundary = phones.rfind(" ", 0, 509)
        if boundary < 1:
            raise ValueError("Text unit exceeds the native token bound")
        chunks.append(phones[:boundary])
        phones = phones[boundary:].lstrip()
    if phones:
        chunks.append(phones)
    return chunks


class KokoroMultispeaker:
    """Load stock once and one or more compatible language adapter bundles.

    Original-stock text inference currently uses American-English af_heart.
    Language adapters select their own finalized fixed speaker conditioning.
    """

    sample_rate = 24000

    def __init__(
        self,
        adapter_dirs=None,
        base_dir=None,
        device="cpu",
        threads=2,
        *,
        languages=None,
    ):
        if threads < 1:
            raise ValueError("threads must be positive")
        if not torch.__version__.startswith("2.6.0"):
            raise RuntimeError("Use the tested PyTorch2.6.0 runtime")
        torch.set_num_threads(threads)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if adapter_dirs is not None and languages is not None:
            raise ValueError(
                "Use languages for downloads or adapter_dirs for local files"
            )
        if adapter_dirs is None:
            registry = json.loads(Path(__file__).with_name("releases.json").read_text())
            if not registry:
                raise ValueError(
                    "No published adapters are configured; supply adapter_dirs"
                )
            selected = tuple(registry) if languages is None else languages
            if (
                not isinstance(selected, (list, tuple))
                or not selected
                or any(
                    not isinstance(lang, str) or lang not in registry
                    for lang in selected
                )
                or len(set(selected)) != len(selected)
            ):
                raise ValueError(
                    "Choose a non-empty list of unique published languages"
                )
            adapter_dirs = {
                language: _default_bundle(language) for language in selected
            }
        if (
            not isinstance(adapter_dirs, dict)
            or not adapter_dirs
            or any(lang not in {"de", "en"} for lang in adapter_dirs)
        ):
            raise ValueError("Supply a non-empty mapping of en/de adapter directories")
        releases = []
        self.configs = {}
        self.speakers = {}
        for language, path in adapter_dirs.items():
            release, config = _checked_bundle(path, language)
            releases.append(release)
            self.configs[language] = config
            self.speakers[language] = tuple(release["speaker_names"])
        if base_dir is None:
            from huggingface_hub import snapshot_download

            base_dir = snapshot_download(
                BASE_REPO,
                revision=BASE_REVISION,
                allow_patterns=["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"],
            )
        root = Path(base_dir)
        for name, digest in [
            ("kokoro-v1_0.pth", BASE_SHA256),
            ("config.json", CONFIG_SHA256),
            ("voices/af_heart.pt", HEART_SHA256),
        ]:
            if sha(root / name) != digest:
                raise ValueError("Stock identity differs: " + name)
        architecture = self.configs[next(iter(self.configs))]["architecture"]
        plbert = self.configs[next(iter(self.configs))]["plbert_config"]
        if any(
            cfg["architecture"] != architecture or cfg["plbert_config"] != plbert
            for cfg in self.configs.values()
        ):
            raise ValueError("Language architecture/configuration differs")
        self.stock = SharedStockKModel(
            repo_id=BASE_REPO,
            config=str(root / "config.json"),
            model=str(root / "kokoro-v1_0.pth"),
        )
        self.stock.to(device).eval().requires_grad_(False)
        for module in list(self.stock.modules()):
            if hasattr(module, "weight_g"):
                torch.nn.utils.remove_weight_norm(module)
        # Use the stock weights as normalized on the inference device.
        native_offsets = {}
        owner = native.Core({name: getattr(self.stock, name) for name in ROOTS})
        graph = native.make_core(architecture, plbert)
        native.fold(graph)
        for name, module in graph.items():
            originals = owner[name].state_dict()
            module.load_state_dict(
                {key: originals[key] for key in module.state_dict()}, strict=True
            )
        self.adapters = SharedAdapters(
            owner,
            releases,
            base_sha256=BASE_SHA256,
            native_weight_offsets=native_offsets,
        )
        self.stock.adapters = self.adapters
        share_stock_graph(owner, graph)
        for module in graph.values():
            module.eval().requires_grad_(False)
        owned = {
            v.data_ptr()
            for module in owner.values()
            for v in module.state_dict().values()
        }
        aliased = {
            v.data_ptr()
            for module in graph.values()
            for v in module.state_dict().values()
        }
        if not aliased <= owned:
            raise RuntimeError("Native graph owns a second backbone")
        self.native_core = graph
        self.device = device
        self.frontends = {}
        self.states = {}
        for release in releases:
            language = release["task"].split("_")[0]
            self.states[language] = {
                "core": graph,
                "voicebank": release["voicebank"].to(device),
                "device": device,
            }
        self.heart = (
            torch.load(
                root / "voices/af_heart.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            .reshape(510, 256)
            .to(device)
        )
        self.storage_verified = True

    def _frontend(self, language):
        key = "en" if language == "stock" else language
        if key not in self.frontends:
            self.frontends[key] = GermanFrontend() if key == "de" else EnglishFrontend()
        return self.frontends[key]

    @torch.inference_mode()
    def controls_tokens(self, tokens, language="de", speaker=None, seed=20261006):
        if (
            not isinstance(tokens, (list, tuple))
            or not 3 <= len(tokens) <= 510
            or tokens[0] != 0
            or tokens[-1] != 0
            or any(
                isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < 178
                for t in tokens
            )
        ):
            raise ValueError("Provide3–510 valid Kokoro tokens with padding boundaries")
        if language == "stock":
            if speaker not in {None, "af_heart"}:
                raise ValueError("Stock text mode currently supports af_heart")
            with self.adapters.selection():
                native.seed(seed)
                ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                wave, duration = self.stock.forward_with_tokens(
                    ids, self.heart[len(tokens) - 3].unsqueeze(0)
                )
                if not torch.isfinite(wave).all():
                    raise RuntimeError("Nonfinite stock output")
                return {
                    "wave": wave.detach().flatten().cpu(),
                    "duration": duration.detach().cpu(),
                }
        if language not in self.states:
            raise ValueError("Language adapter is not loaded: " + str(language))
        names = self.speakers[language]
        speaker = names[0] if speaker is None else speaker
        if speaker not in names:
            raise ValueError("Unknown speaker; available: " + ", ".join(names))
        with self.adapters.selection(language + "_multispeaker"):
            return native.infer(
                self.states[language], tokens, names.index(speaker), seed
            )

    def controls(self, text, language="de", speaker=None, seed=20261006):
        if language != "stock" and language not in self.states:
            raise ValueError("Language adapter is not loaded")
        return self.controls_tokens(
            self._frontend(language).tokens(text), language, speaker, seed
        )

    def stream(
        self,
        text,
        language="de",
        speaker=None,
        seed=20261006,
        *,
        context_before="",
        context_after="",
        stop_event=None,
    ):
        """Yield speech with one neighboring word at each internal boundary.

        Optional external neighbors support successive narration requests. Only
        the body is returned; context audio is discarded using native durations.
        When exact source ownership is unavailable, use production's body-only
        fallback and mark context_used=False.
        """
        text = spoken_text(text, "narration")
        if language != "stock" and language not in self.states:
            raise ValueError("Language adapter is not loaded")
        if language == "stock" and speaker not in {None, "af_heart"}:
            raise ValueError("Stock text mode currently supports af_heart")
        if (
            language != "stock"
            and speaker is not None
            and speaker not in self.speakers[language]
        ):
            raise ValueError(
                "Unknown speaker; available: " + ", ".join(self.speakers[language])
            )
        scoped = prepare_context_text(
            text, context_before, context_after, max_chars=10000
        )
        frontend = self._frontend(language)
        phones, plan, _ = frontend.plan(scoped.text)
        context_range = (
            scoped.owned_words(plan) if plan is not None and scoped.text != text else {}
        )

        def render(window):
            result = self.controls_tokens(model_tokens(window), language, speaker, seed)
            return result["wave"].numpy().reshape(-1), result["duration"].tolist()

        samples = 0
        if plan is not None:
            for part in render_context(
                plan, render, stop_event=stop_event, **context_range
            ):
                samples += len(part.audio)
                if samples > MAX_CHUNK_SAMPLES:
                    raise ValueError(
                        "Synthesis audio exceeds 180 seconds; split the text into shorter passages"
                    )
                yield SpeechChunk(part.audio, part.timings, True)
        else:
            # Exact trimming cannot be proved: never speak optional neighbors.
            if scoped.text != text:
                phones, _, _ = frontend.plan(text)
            for chunk in split_phones(phones):
                if stop_event is not None and stop_event.is_set():
                    return
                audio, _ = render(chunk)
                samples += len(audio)
                if samples > MAX_CHUNK_SAMPLES:
                    raise ValueError(
                        "Synthesis audio exceeds 180 seconds; split the text into shorter passages"
                    )
                yield SpeechChunk(audio, None, False)

    def synthesize(
        self,
        text,
        language="de",
        speaker=None,
        seed=20261006,
        *,
        context_before="",
        context_after="",
    ):
        parts = [
            chunk.audio
            for chunk in self.stream(
                text,
                language,
                speaker,
                seed,
                context_before=context_before,
                context_after=context_after,
            )
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def save(
        self,
        text,
        path,
        language="de",
        speaker=None,
        seed=20261006,
        *,
        context_before="",
        context_after="",
    ):
        import soundfile as sf

        wave = self.synthesize(
            text,
            language,
            speaker,
            seed,
            context_before=context_before,
            context_after=context_after,
        )
        sf.write(path, wave, self.sample_rate, subtype="PCM_16")
        return wave
