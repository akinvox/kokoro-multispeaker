"""Bounded neighboring-word inference shared by AV and stock Kokoro.

The caller retains frontend/model ownership. Only each retained PCM body and
its source-word timings survive an iteration; context PCM is released before
the next model call. No model, queue, cache or network operation lives here.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

CONTEXT_WORDS = 1
BODY_PHONE_LIMIT = 400
SIDE_PHONE_LIMIT = 54
NATIVE_PHONE_LIMIT = 508
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 600
MAX_CHUNK_SAMPLES = SAMPLE_RATE * 180
QUIET_RMS = .006
QUIET_STEP = 120
MIN_QUIET_SAMPLES = 1440
SEARCH_BEFORE_SAMPLES = 15600
SEARCH_AFTER_SAMPLES = 3600
ZERO_SEARCH_SAMPLES = 48
PUNCTUATION = frozenset('.,;:!?"\'()[]{}-\u2014\u2013\u2018\u2019\u201c\u201d')


@dataclass(frozen=True)
class ContextWindow:
    body_start: int
    body_end: int
    window_start: int
    window_end: int
    first_word: int
    last_word: int


@dataclass(frozen=True)
class ContextChunk:
    audio: np.ndarray
    timings: list
    window: ContextWindow


@dataclass(frozen=True)
class ContextText:
    text: str
    body_start: int
    body_end: int

    def owned_words(self, plan):
        """Prove that each optional neighbor owns exactly one source word."""
        owned = [i for i, word in enumerate(plan.words)
                 if self.body_start <= word.start < word.end <= self.body_end]
        if not owned or owned != list(range(owned[0], owned[-1] + 1)):
            raise ValueError('Context body source ownership is unavailable')
        first, end = owned[0], owned[-1] + 1
        if first != int(self.body_start > 0) or len(plan.words) - end != int(self.body_end < len(self.text)):
            raise ValueError('Context must own one source word per side')
        return {'body_start': first, 'body_end': end, 'text_offset': self.body_start}


def prepare_context_text(text, context_before='', context_after='', *, max_chars):
    """Bound optional neighboring text inside the existing provider input limit."""
    sides = []
    for value in (context_before, context_after):
        if not isinstance(value, str) or (value and len(value.split()) != 1):
            raise ValueError('Context must contain at most one word per side')
        sides.append(value.strip())
    before, after = sides
    prefix, suffix = (before + ' ' if before else ''), (' ' + after if after else '')
    combined = prefix + text + suffix
    if len(combined) > max_chars:
        raise ValueError('Text and context exceed the synthesis character limit')
    return ContextText(combined, len(prefix), len(prefix) + len(text))


def word_bounds(plan):
    """Find source-owned phone intervals in one pass, including expansions."""
    if len(plan.phonemes) != len(plan.owners) or not plan.words:
        raise ValueError('Incomplete source phone ownership')
    bounds = [[None, None] for _ in plan.words]
    previous = -1
    for position, owner in enumerate(plan.owners):
        if owner is None:
            continue
        if not isinstance(owner, int) or isinstance(owner, bool) or not previous <= owner < len(bounds):
            raise ValueError('Source phone ownership is not ordered')
        if bounds[owner][0] is None:
            bounds[owner][0] = position
        bounds[owner][1] = position + 1
        previous = owner
    if any(start is None or end is None for start, end in bounds):
        raise ValueError('Source word has no native phones')
    return bounds


def _body_ranges(plan, bounds, first, end):
    while first < end:
        start = bounds[first][0]
        candidates = []
        for last in range(first, end):
            if bounds[last][1] - start > BODY_PHONE_LIMIT:
                break
            candidates.append(last)
        if not candidates:
            if bounds[first][1] - start > NATIVE_PHONE_LIMIT:
                raise ValueError('Text unit exceeds the native token bound')
            last = first
        elif candidates[-1] == end - 1:
            last = candidates[-1]
        else:
            sentence = [i for i in candidates if plan.phonemes[bounds[i][0]:bounds[i][1]].rstrip('"\')]').endswith(('.', '!', '?'))]
            clause = [i for i in candidates if plan.phonemes[bounds[i][0]:bounds[i][1]].rstrip('"\')]').endswith((',', ';', ':'))]
            last = (sentence or clause or candidates)[-1]
        yield first, last
        first = last + 1


def plan_windows(plan, *, body_start=0, body_end=None):
    """Keep whole body words with at most one bounded context word per side."""
    bounds = word_bounds(plan)
    body_end = len(bounds) if body_end is None else body_end
    if (type(body_start) is not int or type(body_end) is not int
            or not 0 <= body_start < body_end <= len(bounds)
            or body_start > CONTEXT_WORDS or len(bounds) - body_end > CONTEXT_WORDS):
        raise ValueError('Body range must exclude at most one context word per side')
    windows = []
    for first, last in _body_ranges(plan, bounds, body_start, body_end):
        body_phone_start, body_phone_end = bounds[first][0], bounds[last][1]
        start, end = body_phone_start, body_phone_end
        for owner in range(first - 1, max(-1, first - CONTEXT_WORDS - 1), -1):
            candidate = bounds[owner][0]
            if body_phone_start - candidate > SIDE_PHONE_LIMIT or end - candidate > NATIVE_PHONE_LIMIT:
                break
            start = candidate
        for owner in range(last + 1, min(len(bounds), last + CONTEXT_WORDS + 1)):
            candidate = bounds[owner][1]
            if candidate - body_phone_end > SIDE_PHONE_LIMIT or candidate - start > NATIVE_PHONE_LIMIT:
                break
            end = candidate
        if start == bounds[0][0]:
            start = 0
        if end == bounds[-1][1]:
            end = len(plan.phonemes)
        if not 0 < end - start <= NATIVE_PHONE_LIMIT:
            raise ValueError('Context exceeds the native token bound')
        windows.append(ContextWindow(body_phone_start, body_phone_end, start, end, first, last))
    return windows, bounds


def quiet_cut(wave, target, minimum, maximum):
    """Prefer actual punctuation silence without entering an adjacent word."""
    left = max(0, minimum, target - SEARCH_BEFORE_SAMPLES)
    right = min(len(wave), maximum, target + SEARCH_AFTER_SAMPLES)
    if left >= right:
        raise ValueError('Source words leave no admitted separator')
    runs, begin = [], None
    for position in range(left, right, QUIET_STEP):
        end = min(position + QUIET_STEP, right)
        level = float(np.sqrt(np.mean(np.square(wave[position:end], dtype=np.float64))))
        if level < QUIET_RMS:
            if begin is None:
                begin = position
        elif begin is not None:
            if position - begin >= MIN_QUIET_SAMPLES:
                runs.append((begin, position))
            begin = None
    if begin is not None and right - begin >= MIN_QUIET_SAMPLES:
        runs.append((begin, right))
    if runs:
        start, end = max(runs, key=lambda pair: (pair[1] - pair[0], -abs(sum(pair) // 2 - target)))
        middle = (start + end) // 2
        lower, upper = max(start + 1, middle - ZERO_SEARCH_SAMPLES), min(end, middle + ZERO_SEARCH_SAMPLES + 1)
    else:
        # Connected speech can have no 60-ms silence. Keep its predicted
        # separator and choose a nearby original sample, never another word.
        middle = min(max(target, left), right - 1)
        lower, upper = max(left, middle - ZERO_SEARCH_SAMPLES), min(right, middle + ZERO_SEARCH_SAMPLES + 1)
    return min(range(lower, upper), key=lambda i: abs(float(wave[i])) + abs(float(wave[max(0, i - 1)])))


def _sample_clock(phones, wave, durations):
    if len(durations) != len(phones) + 2:
        raise ValueError('Native duration token count mismatch')
    if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1 for value in durations):
        raise ValueError('Native durations must be positive integers')
    samples = sum(durations) * SAMPLES_PER_FRAME
    if samples > MAX_CHUNK_SAMPLES or len(wave) != samples or not np.isfinite(wave).all():
        raise ValueError('Native decoder sample clock mismatch or invalid PCM')
    return np.cumsum([0, *durations], dtype=np.int64) * SAMPLES_PER_FRAME


def _separator(plan, bounds, window, clock, before, after):
    first, last = bounds[before][1], bounds[after][0]
    lower, upper = first, last
    while lower > bounds[before][0] and plan.phonemes[lower - 1] in PUNCTUATION:
        lower -= 1
    while upper < bounds[after][1] and plan.phonemes[upper] in PUNCTUATION:
        upper += 1
    offset = 1 - window.window_start
    target = int((clock[offset + first] + clock[offset + last]) // 2)
    return target, int(clock[offset + lower]), int(clock[offset + upper])


def _retained_interval(plan, bounds, window, wave, clock):
    left, right = 0, len(wave)
    if window.first_word > 0 and window.window_start < window.body_start:
        left = quiet_cut(wave, *_separator(plan, bounds, window, clock, window.first_word - 1, window.first_word))
    if window.last_word + 1 < len(bounds) and window.window_end > window.body_end:
        right = quiet_cut(wave, *_separator(plan, bounds, window, clock, window.last_word, window.last_word + 1))
    if not 0 <= left < right <= len(wave):
        raise ValueError('Context trim removed the source body')
    return left, right


def _timings(plan, bounds, window, clock, left, right):
    rows = []
    for owner in range(window.first_word, window.last_word + 1):
        word = plan.words[owner]
        start = max(left, int(clock[1 + bounds[owner][0] - window.window_start]))
        end = min(right, int(clock[1 + bounds[owner][1] - window.window_start]))
        if not left <= start < end <= right:
            raise ValueError('Context trim removed a source word')
        rows.append({'word': word.text, 'start_time': (start - left) / SAMPLE_RATE,
                     'end_time': (end - left) / SAMPLE_RATE, 'duration': (end - start) / SAMPLE_RATE,
                     'text_offset': word.start, 'source_word_index': owner,
                     'boundary_type': 'real_word', 'timing_source': 'kokoro-context-native-sample-clock-v1'})
    return rows


def render_context(plan, render, *, stop_event=None, body_start=0, body_end=None, text_offset=None):
    """Yield retained chunks through a caller-owned model, with source timings."""
    windows, bounds = plan_windows(plan, body_start=body_start, body_end=body_end)
    if text_offset is None:
        text_offset = plan.words[body_start].start if body_start else 0
    for window in windows:
        if stop_event is not None and stop_event.is_set():
            return
        phones = plan.phonemes[window.window_start:window.window_end]
        wave, durations = render(phones)
        wave = np.asarray(wave).reshape(-1)
        clock = _sample_clock(phones, wave, durations)
        left, right = _retained_interval(plan, bounds, window, wave, clock)
        timings = _timings(plan, bounds, window, clock, left, right)
        for row in timings:
            row['source_word_index'] -= body_start
            row['text_offset'] -= text_offset
        retained = wave[left:right].copy()
        del wave, durations, clock
        yield ContextChunk(retained, timings, window)
