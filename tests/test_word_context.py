"""Surrounding context improves delivery without repeating or losing source words."""

import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def context(monkeypatch):
    path = Path(__file__).parents[1] / "kokoro_multispeaker/context.py"
    assert path.is_file(), "The shared bounded Kokoro context implementation is missing"
    spec = importlib.util.spec_from_file_location("kokoro_context_under_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def plan_for(words):
    phones, owners, rows = "", [], []
    for index, word in enumerate(words):
        if index:
            phones += " "
            owners.append(None)
        start = len(phones)
        phones += word
        owners.extend([index] * len(word))
        rows.append(SimpleNamespace(text=word, start=start, end=len(phones)))
    return SimpleNamespace(phonemes=phones, owners=tuple(owners), words=tuple(rows))


@pytest.fixture
def narration():
    return plan_for(
        ["narration" + ("." if index % 15 == 14 else "") for index in range(180)]
    )


def test_windows_include_one_neighboring_word_and_preserve_order(context, narration):
    windows, bounds = context.plan_windows(narration)
    covered = []
    for window in windows:
        first, last = window.first_word, window.last_word
        covered.extend(range(first, last + 1))
        assert window.window_start == bounds[max(0, first - 1)][0]
        assert window.window_end == bounds[min(len(bounds) - 1, last + 1)][1]
        assert window.window_end - window.window_start <= 508
    assert covered == list(range(len(narration.words)))
    assert len(windows) > 1


@pytest.mark.parametrize("ending", [".", ",", ""])
def test_body_cuts_keep_whole_words_and_punctuation(context, ending):
    plan = plan_for(["narration" + ending] * 180)
    windows, bounds = context.plan_windows(plan)
    for window in windows:
        assert window.body_start == bounds[window.first_word][0]
        assert window.body_end == bounds[window.last_word][1]
        assert plan.phonemes[window.body_start : window.body_end].endswith(
            ending or "narration"
        )
    assert "".join(
        plan.phonemes[w.body_start : w.body_end].replace(" ", "") for w in windows
    ) == plan.phonemes.replace(" ", "")


def test_large_words_keep_native_input_bound_without_lowering_it(context):
    plan = plan_for(["hello", "a" * 450, "there", "friend"])
    windows, _ = context.plan_windows(plan)
    assert any(w.body_end - w.body_start == 450 for w in windows)
    assert all(w.window_end - w.window_start <= 508 for w in windows)
    with pytest.raises(ValueError, match="token bound"):
        context.plan_windows(plan_for(["a" * 509]))


def test_word_expansion_internal_spaces_cannot_split_source_identity(context):
    plan = plan_for(["twenty one"] * 85)
    windows, bounds = context.plan_windows(plan)
    assert all(w.body_end == bounds[w.last_word][1] for w in windows)
    assert [i for w in windows for i in range(w.first_word, w.last_word + 1)] == list(
        range(85)
    )


def fake_render(chunk):
    durations = [2] * (len(chunk) + 2)
    wave = np.zeros(sum(durations) * 600, dtype=np.float32)
    for index, char in enumerate(chunk):
        if char.isalnum():
            wave[(index + 1) * 1200 : (index + 2) * 1200] = 0.2
    return wave, durations


def test_context_pcm_is_discarded_and_timings_cover_source_once(context, narration):
    parts = list(context.render_context(narration, fake_render))
    rows = []
    offset = 0
    for part in parts:
        assert (
            part.audio.base is None
        ), "A retained body must release its full context waveform"
        for row in part.timings:
            rows.append(
                dict(
                    row,
                    start_time=row["start_time"] + offset,
                    end_time=row["end_time"] + offset,
                )
            )
        offset += len(part.audio) / 24000
    assert [row["source_word_index"] for row in rows] == list(
        range(len(narration.words))
    )
    assert [row["word"] for row in rows] == [word.text for word in narration.words]
    assert [row["text_offset"] for row in rows] == [
        word.start for word in narration.words
    ]
    assert all(
        a["end_time"] <= b["start_time"] for a, b in zip(rows, rows[1:], strict=False)
    )
    assert all(0 <= row["start_time"] < row["end_time"] <= offset for row in rows)
    expected_spoken_samples = sum(c.isalnum() for c in narration.phonemes) * 1200
    assert (
        sum(np.count_nonzero(part.audio) for part in parts) == expected_spoken_samples
    )


def test_single_chunk_remains_bit_identical(context):
    plan = plan_for(["good", "morning."])
    original, _ = fake_render(plan.phonemes)
    parts = list(context.render_context(plan, fake_render))
    assert len(parts) == 1
    np.testing.assert_array_equal(parts[0].audio, original)


def test_external_one_word_context_is_rendered_but_not_retained(context):
    plan = plan_for(["appeared", "deep", "in", "the", "woods", "behind"])
    calls = []

    def render(phones):
        calls.append(phones)
        return fake_render(phones)

    parts = list(context.render_context(plan, render, body_start=1, body_end=5))

    assert calls == [plan.phonemes]
    rows = [row for part in parts for row in part.timings]
    assert [row["word"] for row in rows] == ["deep", "in", "the", "woods"]
    assert [row["source_word_index"] for row in rows] == list(range(4))
    assert [row["text_offset"] for row in rows] == [0, 5, 8, 12]
    assert (
        sum(np.count_nonzero(part.audio) for part in parts)
        == len("deepinthewoods") * 1200
    )
    assert all(part.audio.base is None for part in parts)


@pytest.mark.parametrize("word_count", [2, 180])
@pytest.mark.parametrize("prefix,suffix", [("", " "), ("(", ""), ("(", "). ")])
def test_unowned_edge_punctuation_keeps_source_audio(
    context, word_count, prefix, suffix
):
    plan = plan_for(["narration"] * word_count)
    plan.phonemes = prefix + plan.phonemes + suffix
    plan.owners = (None,) * len(prefix) + plan.owners + (None,) * len(suffix)

    parts = list(context.render_context(plan, fake_render))

    assert [row["source_word_index"] for part in parts for row in part.timings] == list(
        range(word_count)
    )
    assert (
        sum(np.count_nonzero(part.audio) for part in parts)
        == word_count * len("narration") * 1200
    )
    if word_count == 2:
        original, _ = fake_render(plan.phonemes)
        np.testing.assert_array_equal(parts[0].audio, original)
    else:
        first_wave, _ = fake_render(plan.phonemes[: parts[0].window.window_end])
        last_wave, _ = fake_render(plan.phonemes[parts[-1].window.window_start :])
        np.testing.assert_array_equal(parts[0].audio[:1200], first_wave[:1200])
        np.testing.assert_array_equal(parts[-1].audio[-1200:], last_wave[-1200:])


def test_splice_uses_actual_silence_inside_the_admitted_separator(context):
    wave = np.full(24000, 0.2, dtype=np.float32)
    wave[14000:18000] = 0
    cut = context.quiet_cut(wave, 19500, 13000, 20000)
    assert 14000 <= cut < 18000
    assert abs(wave[cut] - wave[cut - 1]) < 0.005


def test_splice_cannot_jump_to_a_pause_in_a_different_word(context):
    wave = np.full(24000, 0.2, dtype=np.float32)
    wave[1000:8000] = 0
    cut = context.quiet_cut(wave, 15000, 14500, 15500)
    assert 14500 <= cut < 15500


@pytest.mark.parametrize(
    "damage", ["duration_count", "duration_clock", "nonfinite_wave"]
)
def test_bad_predictor_output_fails_without_timing_fabrication(context, damage):
    plan = plan_for(["good", "morning."])
    wave, values = fake_render(plan.phonemes)
    if damage == "duration_count":
        values = values[:-1]
    elif damage == "duration_clock":
        wave = wave[:-1]
    else:
        wave[10] = np.nan
    with pytest.raises(ValueError):
        list(context.render_context(plan, lambda chunk: (wave, values)))


def test_cancelled_stream_stops_before_another_model_call(context, narration):
    stop = threading.Event()
    calls = []

    def render(chunk):
        calls.append(chunk)
        return fake_render(chunk)

    stream = context.render_context(narration, render, stop_event=stop)
    next(stream)
    stop.set()
    assert list(stream) == []
    assert len(calls) == 1


@pytest.mark.parametrize(
    "before,after", [("two words", ""), ("", "two words"), (None, ""), ("", 7)]
)
def test_context_request_rejects_unbounded_or_nontext_neighbors(context, before, after):
    with pytest.raises(ValueError, match="one word"):
        context.prepare_context_text("body", before, after, max_chars=1500)


def test_context_shares_original_character_bound_and_rebases_body_offsets(context):
    scoped = context.prepare_context_text(
        "deep woods", "appeared", "behind", max_chars=26
    )
    plan = plan_for(scoped.text.split())
    options = scoped.owned_words(plan)
    assert options == {"body_start": 1, "body_end": 3, "text_offset": 9}
    rows = [
        row
        for part in context.render_context(plan, fake_render, **options)
        for row in part.timings
    ]
    assert [row["text_offset"] for row in rows] == [0, 5]
    with pytest.raises(ValueError, match="character limit"):
        context.prepare_context_text("deep woods", "appeared", "behind", max_chars=25)


@pytest.mark.parametrize("start,end", [(2, 4), (0, 3), (True, 4), (3, 2), (0, 9)])
def test_renderer_rejects_more_than_one_external_word(context, start, end):
    with pytest.raises(ValueError, match="Body range"):
        list(
            context.render_context(
                plan_for(["one"] * 5), fake_render, body_start=start, body_end=end
            )
        )
