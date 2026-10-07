"""Plain native Kokoro text + voicepack inference; no training helpers."""

import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize
from transformers import AlbertConfig, AlbertModel
from .native_models import TextEncoder, ProsodyPredictor
from .native_decoder import Decoder

ROOTS = ("bert", "bert_encoder", "predictor", "decoder", "text_encoder")


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).last_hidden_state


class Core(dict):
    __getattr__ = dict.__getitem__


def make_core(architecture, plbert_config):
    p = architecture
    dec = dict(p["decoder"])
    assert dec.pop("type") == "istftnet"
    bert = CustomAlbert(AlbertConfig(**plbert_config))
    return Core(
        bert=bert,
        bert_encoder=nn.Linear(bert.config.hidden_size, p["hidden_dim"]),
        text_encoder=TextEncoder(
            channels=p["hidden_dim"],
            kernel_size=5,
            depth=p["n_layer"],
            n_symbols=p["n_token"],
        ),
        predictor=ProsodyPredictor(
            style_dim=p["style_dim"],
            d_hid=p["hidden_dim"],
            nlayers=p["n_layer"],
            max_dur=p["max_dur"],
            dropout=p["dropout"],
        ),
        decoder=Decoder(
            dim_in=p["hidden_dim"], style_dim=p["style_dim"], dim_out=p["n_mels"], **dec
        ),
    )


def fold(core):
    count = 0
    with torch.no_grad():
        for root in core.values():
            for module in root.modules():
                for name in list(getattr(module, "parametrizations", {})):
                    before = getattr(module, name).detach().clone()
                    parametrize.remove_parametrizations(
                        module, name, leave_parametrized=True
                    )
                    assert torch.equal(before, getattr(module, name))
                    count += 1
    return count


def alignment(duration):
    duration = duration.long().flatten()
    if not bool((duration >= 1).all()) or not 0 < int(duration.sum()) <= 4000:
        raise ValueError(
            "Predicted audio exceeds the 4000-frame limit; split the text into shorter sentences"
        )
    owners = torch.repeat_interleave(
        torch.arange(len(duration), device=duration.device), duration
    )
    return (
        torch.nn.functional.one_hot(owners, len(duration))
        .to(torch.float32)
        .T.unsqueeze(0)
    )


def seed(value):
    random.seed(value)
    np.random.seed(value % (2**32 - 1))
    torch.manual_seed(value)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(value)


def safe_wave(wave):
    magnitude = wave.abs()
    compressed = 0.95 + 0.03 * torch.tanh((magnitude - 0.95) / 0.03)
    return torch.where(magnitude <= 0.95, wave, wave.sign() * compressed).clamp(
        -0.98, 0.98
    )


@torch.inference_mode()
def infer(state, token_ids, speaker, seed_value=20260916):
    seed(seed_value)
    assert 3 <= len(token_ids) <= 510
    core, device = state["core"], state["device"]
    text = torch.tensor([token_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(token_ids)], device=device)
    mask = torch.zeros_like(text, dtype=torch.bool)
    base = state["voicebank"][speaker].index_select(0, (lengths - 1).clamp(0, 509))
    s_dec, s_pred = base[:, :128], base[:, 128:]
    bert = core.bert(text, attention_mask=(~mask).int())
    context = core.predictor.text_encoder(
        core.bert_encoder(bert).transpose(-1, -2), s_pred, lengths, mask
    )
    hidden, _ = core.predictor.lstm(context)
    duration_float = core.predictor.duration_proj(hidden).sigmoid().sum(-1).squeeze(0)
    duration = duration_float.round().clamp_min(1).long()
    grid = alignment(duration)
    f0, n = core.predictor.F0Ntrain(context.transpose(-1, -2) @ grid, s_pred)
    content = core.text_encoder(text, lengths, mask) @ grid
    wave = safe_wave(core.decoder(content, f0, n, s_dec).float()).flatten()
    result = dict(
        duration=duration,
        duration_float=duration_float,
        f0=f0,
        n=n,
        s_dec=s_dec,
        s_pred=s_pred,
        content=content,
        wave=wave,
    )
    assert all(torch.isfinite(v).all() for v in result.values())
    return {k: v.detach().cpu() for k, v in result.items()}
