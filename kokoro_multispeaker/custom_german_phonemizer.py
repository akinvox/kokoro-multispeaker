#!/usr/bin/env python3
"""German eSpeak frontend matching the released model vocabulary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from phonemizer.backend import EspeakBackend

# The German adapters were trained on eSpeak-style IPA rather than Kikiri's
# newer DEG2P compact-phone convention. Keep that convention, but do not keep
# the historical silent-OOV behavior.
GERMAN_IPA_REPLACEMENTS = {
    "??": "ʊɐ",  # eSpeak-NG 1.50 data artifact observed for "wurde".
    "ʏ": "y",  # Kokoro has y but no U+028F short-u symbol.
    # eSpeak-NG 1.50 emits a literal non-IPA ``1`` in standalone
    # "aneinander" (``ˌan1aɪnˈandɜ``), while the same morphemes in
    # "aneinanderreihen" are emitted correctly as ``ˈanaɪnˌandɜ...``.
    # Removing the artifact preserves the intended phones and keeps the
    # Kokoro-178 vocabulary strict instead of silently dropping the token.
    "1": "",
    # eSpeak-NG 1.50 can move the stress in the English loanword "Updates"
    # after unrelated language-switching calls in the same process.  The
    # release-pure training lists were sealed with the fresh-process form;
    # make text-only inference resolve the alternate form identically.
    "ʌpdˈeɪts": "ˈʌpdeɪts",
}


@dataclass(frozen=True)
class Phonemization:
    text: str
    ipa: str
    token_ids: tuple[int, ...]


class StrictGermanEspeak:
    """Raw German eSpeak IPA with deterministic Kokoro-178 validation."""

    def __init__(self, vocabulary: dict[str, int]):
        self.vocabulary = dict(vocabulary)
        self.backend = EspeakBackend(
            "de",
            language_switch="remove-flags",
            with_stress=True,
            preserve_punctuation=True,
        )

    @staticmethod
    def normalize(ipa: str) -> str:
        # Canonicalize all Unicode whitespace (including NBSP observed in one
        # Thorsten transcript), not only literal ASCII spaces/newlines.
        normalized = re.sub(r"\s+", " ", ipa).strip()
        for source, target in GERMAN_IPA_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)
        return re.sub(r"\s+", " ", normalized).strip()

    def phonemize_many(self, texts: Iterable[str]) -> list[Phonemization]:
        texts = list(texts)
        raw_phonemes = self.backend.phonemize(texts)
        if len(raw_phonemes) != len(texts):
            raise RuntimeError(
                f"eSpeak result count mismatch: {len(raw_phonemes)} != {len(texts)}"
            )

        results = []
        for text, raw_ipa in zip(texts, raw_phonemes):
            ipa = self.normalize(raw_ipa)
            unknown = sorted(set(ipa).difference(self.vocabulary))
            if unknown:
                formatted = ", ".join(
                    f"{char!r} (U+{ord(char):04X})" for char in unknown
                )
                raise ValueError(f"Unsupported German IPA for {text!r}: {formatted}")
            if not ipa or len(ipa) > 510:
                raise ValueError(f"Invalid IPA length {len(ipa)} for {text!r}")
            token_ids = tuple(self.vocabulary[character] for character in ipa)
            if not token_ids or min(token_ids) < 0 or max(token_ids) >= 178:
                raise ValueError(f"Invalid Kokoro token IDs for {text!r}")
            results.append(Phonemization(text=text, ipa=ipa, token_ids=token_ids))
        return results

    def __call__(self, text: str) -> Phonemization:
        first = self.phonemize_many([text])[0]
        second = self.phonemize_many([text])[0]
        if first != second:
            raise RuntimeError(f"Non-deterministic German phonemization for {text!r}")
        return first


def frontend_manifest() -> dict:
    return {
        "name": "strict-custom-espeak-de-kokoro178-v1",
        "backend": "phonemizer.backend.EspeakBackend",
        "language": "de",
        "language_switch": "remove-flags",
        "with_stress": True,
        "preserve_punctuation": True,
        "replacements": GERMAN_IPA_REPLACEMENTS,
        "silent_oov_drop": False,
    }
