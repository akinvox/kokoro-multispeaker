# Changes: public module location; synthetic clone task becomes a language task; clone-specific tests omitted.
"""Stock and native math must share the same tensors, including selected deltas."""

import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn

SPEC = importlib.util.spec_from_file_location(
    "shared_kokoro_graph",
    Path(__file__).parents[1] / "kokoro_multispeaker/language_weights.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
BASE = "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4"


def share(owner, graph):
    assert hasattr(
        module, "share_stock_graph"
    ), "Stock/native math still allocates independent weights"
    module.share_stock_graph(owner, graph)


def test_native_graph_reuses_original_parameters_and_identical_adapter_inventory():
    owner = {
        "bert": nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        .eval()
        .requires_grad_(False)
    }
    graph = {
        "bert": nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        .eval()
        .requires_grad_(False)
    }
    release = {
        "task": "fr_multispeaker",
        "base_sha256": BASE,
        "targets": [
            {
                "root": "bert",
                "module": "0",
                "parameter": "weight",
                "rank": 2,
                "alpha": 2.0,
                "prefix": "bert.0.weight",
            }
        ],
        "state": {
            "bert.0.weight.A": torch.ones(2, 4),
            "bert.0.weight.B": torch.ones(3, 2),
        },
    }
    shared = module.SharedAdapters(owner, [release], base_sha256=BASE)
    share(owner, graph)
    assert (
        graph["bert"][0].parametrizations.weight
        is owner["bert"][0].parametrizations.weight
    )
    assert graph["bert"][0].bias is owner["bert"][0].bias
    assert graph["bert"][1].weight is owner["bert"][1].weight
    sample = torch.ones(1, 4)
    with shared.selection("fr_multispeaker"):
        assert torch.equal(owner["bert"](sample), graph["bert"](sample))
    assert torch.equal(owner["bert"](sample), graph["bert"](sample))
    assert {v.data_ptr() for v in graph["bert"].state_dict().values()} <= {
        v.data_ptr() for v in owner["bert"].state_dict().values()
    }


def test_graph_preserves_nonaffine_native_normalization_without_changing_stock_modules():
    owner = {"predictor": nn.Sequential(nn.InstanceNorm1d(3, affine=True))}
    graph = {"predictor": nn.Sequential(nn.InstanceNorm1d(3, affine=False))}
    share(owner, graph)
    assert owner["predictor"][0].affine and not graph["predictor"][0].affine
    assert graph["predictor"][0].weight is None


def test_constant_buffers_are_shared_instead_of_copied():
    owner, graph = {"bert": nn.Module()}, {"bert": nn.Module()}
    for root in (owner, graph):
        root["bert"].register_buffer("constant", torch.ones(3))
    share(owner, graph)
    assert graph["bert"].constant is owner["bert"].constant


def test_incompatible_graph_is_rejected_before_replacing_any_weights():
    owner = {"bert": nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))}
    graph = {"bert": nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 5))}
    before = graph["bert"][0].weight
    with pytest.raises(ValueError, match="shape"):
        share(owner, graph)
    assert graph["bert"][0].weight is before


def test_folded_native_weight_buffer_shares_the_original_selected_parameter():
    owner = {"bert": nn.Linear(4, 3, bias=False).eval().requires_grad_(False)}
    graph = {"bert": nn.Module()}
    graph["bert"].register_buffer("weight", torch.zeros(3, 4))
    release = {
        "task": "fr_multispeaker",
        "base_sha256": BASE,
        "targets": [
            {
                "root": "bert",
                "module": "",
                "parameter": "weight",
                "rank": 2,
                "alpha": 2.0,
                "prefix": "bert.weight",
            }
        ],
        "state": {"bert.weight.A": torch.ones(2, 4), "bert.weight.B": torch.ones(3, 2)},
    }
    shared = module.SharedAdapters(owner, [release], base_sha256=BASE)
    share(owner, graph)
    assert (
        graph["bert"].parametrizations.weight is owner["bert"].parametrizations.weight
    )
    with shared.selection("fr_multispeaker"):
        assert torch.equal(graph["bert"].weight, owner["bert"].weight)
