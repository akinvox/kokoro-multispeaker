"""Select one inference LoRA over immutable stock Kokoro weights.

Each request selects one adapter. Stock mode uses the original weights.
A lock holds that selection until inference finishes.
"""

import math
import re
import threading
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn.utils import parametrize

ROOTS = frozenset(("bert", "bert_encoder", "predictor", "decoder", "text_encoder"))
MAX_TARGETS = 201
MAX_ADAPTER_BYTES = 100 * 1024**2
MAX_NATIVE_NORMALIZATION_BYTES = 96 * 1024**2
MAX_RANK = 64
PARAMETER = re.compile("weight(?:_(?:ih|hh)_l[0-9]+(?:_reverse)?)?\\Z")


class _LowRankDelta(nn.Module):

    def __init__(self, a, b, shape, alpha):
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.shape = tuple(shape)
        self.scale = float(alpha) / a.shape[0]

    def forward(self, original):
        return (
            original
            + (self.b @ self.a).reshape(self.shape).to(original.dtype) * self.scale
        )


class _TaskWeights(nn.Module):

    def __init__(self, owner, deltas, native_offsets=None):
        super().__init__()
        self.owner = owner
        self.deltas = nn.ModuleDict(deltas)
        self.register_buffer("native_offsets", native_offsets)

    def native_original(self, original):
        if self.native_offsets is None:
            return original
        return (original.view(torch.int32) + self.native_offsets.to(torch.int32)).view(
            torch.float32
        )

    def forward(self, original):
        task = self.owner.active
        if task is None:
            return original
        original = self.native_original(original)
        result = self.deltas[task](original) if task in self.deltas else original
        return result


def _target(core, target, state):
    root, module_name = (target["root"], target["module"])
    name, prefix = (target["parameter"], target["prefix"])
    if root not in ROOTS or root not in core or (not isinstance(module_name, str)):
        raise ValueError("Unknown adapter root or module")
    if not isinstance(name, str) or not PARAMETER.fullmatch(name):
        raise ValueError("Unsupported adapter parameter")
    module = core[root].get_submodule(module_name) if module_name else core[root]
    original = getattr(module, name)
    if parametrize.is_parametrized(module, name) or not isinstance(
        original, nn.Parameter
    ):
        raise ValueError("Shared backbone weights must be folded stock parameters")
    ak, bk = (prefix + ".A", prefix + ".B")
    a, b = (state[ak], state[bk])
    rank, alpha = (target["rank"], target["alpha"])
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.ndim != 2
        or (b.ndim != 2)
        or (original.ndim < 2)
        or isinstance(rank, bool)
        or (not isinstance(rank, int))
        or (not 1 <= rank <= MAX_RANK)
        or (a.shape[0] != rank)
        or (b.shape[1] != rank)
        or (b.shape[0] != original.shape[0])
        or (a.shape[1] * b.shape[0] != original.numel())
        or (a.dtype != original.dtype)
        or (b.dtype != original.dtype)
        or isinstance(alpha, bool)
        or (not isinstance(alpha, (int, float)))
        or (not math.isfinite(alpha))
        or (not 0 < alpha <= MAX_RANK)
        or (not torch.isfinite(a).all())
        or (not torch.isfinite(b).all())
    ):
        raise ValueError("Invalid low-rank tensor: " + prefix)
    return (module, name, a, b, original, alpha, {ak, bk})


def _validated_targets(core, release, base_sha256):
    if (
        not isinstance(release.get("task"), str)
        or not re.fullmatch("[a-z]{2,3}_multispeaker", release["task"])
        or release.get("base_sha256") != base_sha256
    ):
        raise ValueError("Adapter task or immutable base identity differs")
    targets, state = (release.get("targets"), release.get("state"))
    if (
        not isinstance(targets, list)
        or not 1 <= len(targets) <= MAX_TARGETS
        or (not isinstance(state, dict))
    ):
        raise ValueError("Invalid adapter target inventory")
    consumed, seen, result, total = (set(), set(), [], 0)
    for target in targets:
        try:
            item = _target(core, target, state)
        except (KeyError, AttributeError, TypeError) as error:
            raise ValueError("Invalid adapter target or tensor inventory") from error
        module, name, a, b, _, _, keys = item
        identity = (id(module), name)
        if identity in seen or keys & consumed:
            raise ValueError("Duplicate adapter target")
        total += a.numel() * a.element_size() + b.numel() * b.element_size()
        if total > MAX_ADAPTER_BYTES:
            raise ValueError("Adapter tensors exceed the fixed byte budget")
        seen.add(identity)
        consumed.update(keys)
        result.append(item)
    if consumed != set(state):
        raise ValueError("Adapter state contains unconsumed tensors")
    return result


class SharedAdapters:
    """Select a declared language adapter over one immutable stock weight owner."""

    def __init__(self, core, releases, *, base_sha256, native_weight_offsets=None):
        if not isinstance(base_sha256, str) or not re.fullmatch(
            "[0-9a-f]{64}", base_sha256
        ):
            raise ValueError("Invalid immutable stock base identity")
        if not isinstance(releases, list) or not releases:
            raise ValueError("Invalid shared adapter inventory")
        self.active = None
        self._lock = threading.RLock()
        self.tasks = set()
        grouped = {}
        for release in releases:
            validated = _validated_targets(core, release, base_sha256)
            task = release["task"]
            if task in self.tasks:
                raise ValueError("Duplicate adapter task")
            self.tasks.add(task)
            for target, (module, name, a, b, original, alpha, _) in zip(
                release["targets"], validated, strict=True
            ):
                identity = (id(module), name)
                group = grouped.setdefault(identity, (module, name, {}))
                group[2][task] = _LowRankDelta(
                    a.to(original.device), b.to(original.device), original.shape, alpha
                )
        corrections, total = ({}, 0)
        for (root, path, name), offsets in (native_weight_offsets or {}).items():
            if (
                root not in ROOTS
                or root not in core
                or (not isinstance(path, str))
                or (name != "weight")
            ):
                raise ValueError("Unknown native normalization target")
            module = core[root].get_submodule(path) if path else core[root]
            original = getattr(module, name)
            if (
                not isinstance(original, nn.Parameter)
                or original.dtype != torch.float32
                or (not original.is_contiguous())
                or (not isinstance(offsets, torch.Tensor))
                or (offsets.dtype != torch.int8)
                or (offsets.shape != original.shape)
            ):
                raise ValueError("Invalid native normalization offsets")
            total += offsets.numel()
            if total > MAX_NATIVE_NORMALIZATION_BYTES:
                raise ValueError("Native normalization exceeds its fixed byte budget")
            identity = (id(module), name)
            grouped.setdefault(identity, (module, name, {}))
            corrections[identity] = offsets.to(original.device)
        self.normalization_bytes = total
        prepared = []
        for identity, (module, name, deltas) in grouped.items():
            prepared.append(
                (
                    identity,
                    module,
                    name,
                    _TaskWeights(self, deltas, corrections.get(identity)),
                )
            )
        for _, module, name, weights in prepared:
            parametrize.register_parametrization(module, name, weights)
            module.requires_grad_(False)

    @contextmanager
    def selection(self, task=None):
        """Serialize task selection; restore the previous state even on failure."""
        if task is not None and task not in self.tasks:
            raise ValueError("Unknown shared adapter task")
        with self._lock:
            previous, self.active = (self.active, task)
            try:
                yield
            finally:
                self.active = previous


def share_stock_graph(owner, graph):
    """Bind native math to the stock owner's exact parameters and adapter banks.

    Native and released stock kernels differ. Keep their small Python graphs,
    while every native parameter and constant aliases its single stock owner.
    Build/fold the native graph on CPU; initialize the owner on its final device
    first. No second backbone or adapter tensors are moved to the GPU.
    """
    if set(owner) != set(graph) or not set(owner) <= ROOTS:
        raise ValueError("Shared graph roots differ")
    actions = []
    for root, network in graph.items():
        for path, child in list(network.named_modules()):
            if getattr(child, "parametrizations", None):
                raise ValueError(
                    "Native graph must be folded before binding stock weights"
                )
            try:
                original = owner[root].get_submodule(path) if path else owner[root]
            except AttributeError as error:
                raise ValueError("Native graph module is absent from stock") from error
            for name, value in child.named_parameters(recurse=False):
                candidate = getattr(original, name, None)
                if (
                    candidate is None
                    or candidate.shape != value.shape
                    or candidate.dtype != value.dtype
                ):
                    raise ValueError("Shared graph parameter shape or dtype differs")
                is_adapter = parametrize.is_parametrized(original, name)
                target = original.parametrizations[name] if is_adapter else candidate
                actions.append((child, name, target, is_adapter))
            for name, value in child.named_buffers(recurse=False):
                candidate = getattr(original, name, None)
                if (
                    candidate is None
                    or candidate.shape != value.shape
                    or candidate.dtype != value.dtype
                ):
                    raise ValueError(
                        "Shared graph constant shape, dtype or value differs: "
                        + ".".join((part for part in (root, path, name) if part))
                    )
                if parametrize.is_parametrized(original, name):
                    actions.append((child, name, original.parametrizations[name], True))
                elif isinstance(candidate, nn.Parameter):
                    actions.append((child, name, candidate, None))
                else:
                    if not torch.equal(value.to(candidate.device), candidate):
                        raise ValueError("Shared graph constant value differs")
                    actions.append((child, name, candidate, None))
    for child, name, target, is_adapter in actions:
        if is_adapter is None:
            child._buffers[name] = target
        elif is_adapter:
            parametrize.register_parametrization(child, name, nn.Identity())
            child.parametrizations[name] = target
        else:
            child._parameters[name] = target
