# Changes: public module location; synthetic clone task becomes a language task; clone-specific tests omitted.
"""Stock integrity and task isolation for the single Kokoro backbone."""

import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

MODULE = Path(__file__).parents[1] / "kokoro_multispeaker/language_weights.py"
BASE = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"


def implementation():
    assert MODULE.is_file(), "Shared stock adapter selection has not been implemented"
    spec = importlib.util.spec_from_file_location("shared_kokoro_adapters", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def release(task, *, parameter="weight", shape=(3, 4), offset=1):
    rank = 2
    prefix = "bert." + parameter
    return {
        "task": task,
        "base_sha256": BASE,
        "targets": [
            {
                "root": "bert",
                "module": "",
                "parameter": parameter,
                "rank": rank,
                "alpha": 2.0,
                "prefix": prefix,
            }
        ],
        "state": {
            prefix + ".A": torch.full((rank, shape[1]), float(offset)),
            prefix + ".B": torch.ones(shape[0], rank),
        },
    }


def setup():
    core = {"bert": nn.Linear(4, 3, bias=False).eval().requires_grad_(False)}
    original = core["bert"].weight.detach().clone()
    tasks = [
        release(task, offset=i)
        for i, task in enumerate(
            ("fr_multispeaker", "de_multispeaker", "en_multispeaker"), start=1
        )
    ]
    shared = implementation().SharedAdapters(core, tasks, base_sha256=BASE)
    return core, original, shared


def test_stock_is_exact_before_after_repeated_task_switches():
    core, original, shared = setup()
    sample = torch.arange(4, dtype=torch.float32).unsqueeze(0)
    expected = sample @ original.T
    assert torch.equal(core["bert"](sample), expected)
    for _ in range(3):
        for index, task in enumerate(
            ("fr_multispeaker", "de_multispeaker", "en_multispeaker"), 1
        ):
            with shared.selection(task):
                assert torch.equal(core["bert"].weight, original + 2 * index)
                assert torch.equal(
                    core["bert"](sample), sample @ (original + 2 * index).T
                )
            assert torch.equal(core["bert"](sample), expected)
    assert torch.equal(core["bert"].parametrizations.weight.original, original)


def test_exception_restores_stock_and_releases_selection():
    core, original, shared = setup()
    with pytest.raises(RuntimeError, match="owned inference failed"):
        with shared.selection("de_multispeaker"):
            raise RuntimeError("owned inference failed")
    assert torch.equal(core["bert"].weight, original)
    with shared.selection("fr_multispeaker"):
        assert torch.equal(core["bert"].weight, original + 2)


def test_simultaneous_requests_cannot_mix_adapters_or_stock():
    core, original, shared = setup()
    entered, release_first, second_waiting = (threading.Event() for _ in range(3))

    def first():
        with shared.selection("de_multispeaker"):
            entered.set()
            assert release_first.wait(3)
            return core["bert"].weight.detach().clone()

    def second():
        assert entered.wait(3)
        second_waiting.set()
        with shared.selection(None):
            return core["bert"].weight.detach().clone()

    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.submit(first), executor.submit(second)
        assert second_waiting.wait(3)
        assert not b.done()
        release_first.set()
        assert torch.equal(a.result(3), original + 4)
        assert torch.equal(b.result(3), original)


def test_task_missing_a_target_uses_stock_for_that_parameter():
    module = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
    core = {"bert": module.eval().requires_grad_(False)}
    tasks = [release("fr_multispeaker"), release("de_multispeaker", shape=(2, 3))]
    tasks[0]["targets"][0]["module"] = "0"
    tasks[1]["targets"][0]["module"] = "1"
    originals = [part.weight.detach().clone() for part in module]
    shared = implementation().SharedAdapters(core, tasks, base_sha256=BASE)
    with shared.selection("fr_multispeaker"):
        assert torch.equal(module[0].weight, originals[0] + 2)
        assert torch.equal(module[1].weight, originals[1])
    with shared.selection("de_multispeaker"):
        assert torch.equal(module[0].weight, originals[0])
        assert torch.equal(module[1].weight, originals[1] + 2)


def test_recurrent_weights_follow_the_selected_task():
    lstm = nn.LSTM(4, 3, batch_first=True).eval().requires_grad_(False)
    core = {"bert": lstm}
    task = release("en_multispeaker", parameter="weight_ih_l0", shape=(12, 4))
    original = lstm.weight_ih_l0.detach().clone()
    shared = implementation().SharedAdapters(core, [task], base_sha256=BASE)
    with shared.selection("en_multispeaker"):
        assert torch.equal(lstm.weight_ih_l0, original + 2)
        assert torch.isfinite(lstm(torch.ones(1, 3, 4))[0]).all()
    assert torch.equal(lstm.weight_ih_l0, original)


@pytest.mark.parametrize(
    "failure", ["base", "shape", "finite", "extra", "duplicate", "task"]
)
def test_invalid_release_is_rejected_before_any_core_mutation(failure):
    core = {"bert": nn.Linear(4, 3, bias=False).eval().requires_grad_(False)}
    first, bad = release("fr_multispeaker"), release("de_multispeaker")
    if failure == "base":
        bad["base_sha256"] = "0" * 64
    elif failure == "shape":
        bad["state"]["bert.weight.B"] = torch.zeros(4, 2)
    elif failure == "finite":
        bad["state"]["bert.weight.A"][0, 0] = float("nan")
    elif failure == "extra":
        bad["state"]["unused"] = torch.ones(1)
    elif failure == "duplicate":
        bad["targets"] *= 2
    elif failure == "task":
        bad["task"] = "unsealed_task"
    with pytest.raises(ValueError):
        implementation().SharedAdapters(core, [first, bad], base_sha256=BASE)
    assert not parametrize.is_parametrized(core["bert"])


def test_unknown_task_does_not_change_current_selection():
    core, original, shared = setup()
    with pytest.raises(ValueError, match="Unknown"):
        with shared.selection("unsealed_task"):
            pytest.fail("Invalid task admitted")
    assert torch.equal(core["bert"].weight, original)


def test_no_effective_weight_copies_are_retained_after_inference():
    core, original, shared = setup()
    baseline = sum(
        t.numel() * t.element_size() for t in core["bert"].state_dict().values()
    )
    for _ in range(5):
        with shared.selection("fr_multispeaker"):
            core["bert"](torch.ones(1, 4))
    assert (
        sum(t.numel() * t.element_size() for t in core["bert"].state_dict().values())
        == baseline
    )
    assert torch.equal(core["bert"].parametrizations.weight.original, original)


def test_stock_and_native_keep_their_exact_normalization_rounding():
    module = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
    module.eval().requires_grad_(False)
    core = {"bert": module}
    originals = [part.weight.detach().clone() for part in module]
    native = [
        torch.nextafter(value, torch.full_like(value, float("inf")))
        for value in originals
    ]
    offsets = {
        ("bert", str(index), "weight"): (
            value.view(torch.int32) - original.view(torch.int32)
        ).to(torch.int8)
        for index, (original, value) in enumerate(zip(originals, native, strict=True))
    }
    task = release("en_multispeaker", offset=0)
    task["targets"][0]["module"] = "0"
    shared = implementation().SharedAdapters(
        core, [task], base_sha256=BASE, native_weight_offsets=offsets
    )
    before = sum(
        value.numel() * value.element_size() for value in module.state_dict().values()
    )
    for _ in range(3):
        assert all(
            torch.equal(part.weight, value)
            for part, value in zip(module, originals, strict=True)
        )
        with shared.selection("en_multispeaker"):
            assert all(
                torch.equal(part.weight, value)
                for part, value in zip(module, native, strict=True)
            )
    assert all(
        torch.equal(part.weight, value)
        for part, value in zip(module, originals, strict=True)
    )
    assert (
        sum(
            value.numel() * value.element_size()
            for value in module.state_dict().values()
        )
        == before
    )
