"""German phonemization with an explicit eSpeak-NG 1.51 requirement."""

import os
import re
from pathlib import Path
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from .custom_german_phonemizer import StrictGermanEspeak
from .kokoro_symbols import dicts
from .word_plan import TimingUnavailable, english_plan, german_plan


def spoken_text(text, language):
    if (
        not isinstance(text, str)
        or not text.strip()
        or not any(c.isalnum() for c in text)
    ):
        raise ValueError(
            f"Provide non-empty {language} text with spoken words or numbers"
        )
    return text.strip()


def model_tokens(phones):
    tokens = [0, *[dicts[phone] for phone in phones], 0]
    if len(tokens) > 510:
        raise ValueError(
            "Text exceeds 510 model tokens; use synthesize or stream for longer text"
        )
    return tokens


class GermanFrontend:
    def __init__(self):
        library = os.environ.get("KOKORO_ESPEAK_LIBRARY")
        data = os.environ.get("KOKORO_ESPEAK_DATA")
        if library:
            if not Path(library).is_file():
                raise ValueError("KOKORO_ESPEAK_LIBRARY must name an existing library")
            EspeakWrapper.set_library(library)
        if data:
            if not Path(data).is_dir():
                raise ValueError(
                    "KOKORO_ESPEAK_DATA must name an existing data directory"
                )
            EspeakWrapper.set_data_path(data)
        version = EspeakWrapper().version
        if tuple(version[:2]) != (1, 51):
            raise RuntimeError(
                f"This release requires eSpeak-NG 1.51; found {version}. See GUIDE.md for installation."
            )
        self.backend = StrictGermanEspeak(dicts)

    def phonemes(self, text):
        text = spoken_text(text, "German")
        first = self.backend.backend.phonemize([text])
        second = self.backend.backend.phonemize([text])
        if len(first) != 1 or len(second) != 1:
            raise RuntimeError("German frontend result count mismatch")
        phones = self.backend.normalize(first[0])
        if phones != self.backend.normalize(second[0]):
            raise RuntimeError("Non-deterministic German phonemization")
        if not phones or any(phone not in dicts for phone in phones):
            raise ValueError("Unsupported German phonemes")
        return phones

    def tokens(self, text):
        return model_tokens(self.phonemes(text))

    def plan(self, text):
        phones = self.phonemes(text)
        try:
            parts = self.backend.backend.phonemize(re.findall(r"\S+", text))
            parts = [self.backend.normalize(value) for value in parts]
            return phones, german_plan(text, phones, parts), None
        except TimingUnavailable as error:
            return phones, None, str(error)
        except Exception as error:
            return (
                phones,
                None,
                "espeak_timing_observation_failed:" + type(error).__name__,
            )


class EnglishFrontend:
    """American-English Misaki frontend and punctuation mapping."""

    def __init__(self):
        from misaki.en import G2P
        from misaki.espeak import EspeakFallback

        self.backend = G2P(
            trf=False, british=False, fallback=EspeakFallback(british=False), unk=""
        )

    def _phonemes(self, text):
        text = spoken_text(text, "English")
        raw, words = self.backend(text)
        phones = raw.translate(str.maketrans({"[": "(", "]": ")"}))
        if not phones or any(p not in dicts for p in phones):
            raise ValueError("Unsupported English phonemes")
        return raw, phones, words

    def tokens(self, text):
        return model_tokens(self._phonemes(text)[1])

    def plan(self, text):
        raw, phones, words = self._phonemes(text)
        try:
            return phones, english_plan(text, raw, words), None
        except TimingUnavailable as error:
            return phones, None, str(error)
