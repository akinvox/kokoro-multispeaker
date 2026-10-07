"""Source-word ownership for the unchanged Misaki/eSpeak phoneme sequence."""

from dataclasses import dataclass
import re


def canonicalize_punctuation(phones):
    return phones.translate(str.maketrans({"[": "(", "]": ")"}))


STRESS = frozenset("ˈˌ")


class TimingUnavailable(ValueError):
    """The existing audio is valid but this source mapping is not proven."""


@dataclass(frozen=True)
class Word:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class PhonePlan:
    phonemes: str
    owners: tuple
    words: tuple
    provenance: str


def source_words(text):
    # EN/DE app timing coordinates are whitespace groups with an alphanumeric.
    return tuple(
        Word(m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
        if any(c.isalnum() for c in m.group())
    )


def _plan(phonemes, owners, words, provenance):
    if len(phonemes) != len(owners) or not words:
        raise TimingUnavailable("source_phone_ownership_incomplete")
    covered = {owner for owner in owners if owner is not None}
    if covered != set(range(len(words))):
        raise TimingUnavailable("source_word_has_no_native_phones")
    previous = -1
    for owner in owners:
        if owner is None:
            continue
        if owner < previous:
            raise TimingUnavailable("source_word_ownership_interleaved")
        previous = owner
    return PhonePlan(phonemes, tuple(owners), words, provenance)


def english_plan(text, raw_phonemes, tokens):
    """Preserve the actual Misaki result and its original-text token ownership."""
    words = source_words(text)
    reconstructed = "".join((t.phonemes or "") + t.whitespace for t in tokens)
    original_text = "".join(t.text + t.whitespace for t in tokens)
    if reconstructed != raw_phonemes or original_text != text.lstrip():
        raise TimingUnavailable("misaki_source_or_phone_reconstruction_mismatch")
    owners = []
    cursor = len(text) - len(text.lstrip())
    for token in tokens:
        end = cursor + len(token.text)
        overlap = [
            i for i, word in enumerate(words) if word.start < end and word.end > cursor
        ]
        if len(overlap) > 1:
            raise TimingUnavailable("misaki_token_spans_multiple_source_words")
        owner = overlap[0] if overlap else None
        owners.extend([owner] * len(token.phonemes or ""))
        owners.extend([None] * len(token.whitespace))
        cursor = end + len(token.whitespace)
    return _plan(
        canonicalize_punctuation(raw_phonemes),
        owners,
        words,
        "misaki_actual_token_source_ownership",
    )


def german_plan(text, phonemes, word_phonemes):
    """Verify source decomposition without modifying whole-utterance eSpeak IPA.

    Context changes stress markers in eSpeak's isolated word observations. They
    are ignored only while matching source ownership; every original marker
    remains in the model input and receives its actual predicted duration.
    Any other difference rejects timing. Number expansions keep all internal
    phoneme spaces within their original source word.
    """
    groups = list(re.finditer(r"\S+", text))
    if len(groups) != len(word_phonemes):
        raise TimingUnavailable("espeak_source_group_count_mismatch")
    words = source_words(text)
    indices = {(w.start, w.end): i for i, w in enumerate(words)}
    expected, reference_owners = [], []
    for position, (group, part) in enumerate(zip(groups, word_phonemes)):
        if position:
            expected.append(" ")
            reference_owners.append(None)
        owner = indices.get((group.start(), group.end()))
        for char in part:
            if char not in STRESS:
                expected.append(char)
                reference_owners.append(owner)
    reference = "".join(expected)
    # Empty punctuation observations may leave repeated separators; normalize
    # precisely the whitespace transformation already used by the DE frontend.
    normalized_chars, normalized_owners = [], []
    for char, owner in zip(reference, reference_owners):
        if char.isspace():
            if normalized_chars and normalized_chars[-1] != " ":
                normalized_chars.append(" ")
                normalized_owners.append(owner)
        else:
            normalized_chars.append(char)
            normalized_owners.append(owner)
    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        normalized_owners.pop()
    observed = "".join(c for c in phonemes if c not in STRESS)
    if observed != "".join(normalized_chars):
        raise TimingUnavailable("espeak_nonstress_source_decomposition_mismatch")
    owners, offset = [], 0
    for char in phonemes:
        if char in STRESS:
            # A stress marker belongs to the immediately following native
            # phone; it never substitutes for a missing source phone.
            if offset >= len(normalized_owners) or normalized_chars[offset] == " ":
                raise TimingUnavailable("espeak_stress_marker_has_no_word_owner")
            owners.append(normalized_owners[offset])
        else:
            owners.append(normalized_owners[offset])
            offset += 1
    return _plan(
        phonemes,
        owners,
        words,
        "espeak_exact_nonstress_source_decomposition_original_stress_retained",
    )
