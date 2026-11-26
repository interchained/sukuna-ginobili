import torch
import torch.nn as nn

from .config import GinobiliConfig
from .experts import ExpertBank


class GraftedMLP(nn.Module):
    """y_stage(z) = F_frozen(z) + ExpertBank(z).

    Wraps one decoder layer's frozen MLP with a parallel expert branch.
    The host layer adds its residual around this module, giving exactly
    y = x + F_frozen(z) + sum_{i in TopK} p_i * E_i(z).
    """

    def __init__(self, base_mlp: nn.Module, bank: ExpertBank):
        super().__init__()
        self.base_mlp = base_mlp
        self.bank = bank
        self.last_routing = None

    def forward(self, z: torch.Tensor, **kwargs):
        branch, routing = self.bank(z)
        self.last_routing = routing
        return self.base_mlp(z) + branch


def graft_model(model: nn.Module, config: GinobiliConfig | None = None) -> nn.Module:
    """Graft an ExpertBank onto every decoder layer of a Llama-style model.

    Freezes every backbone parameter; only routers and experts train.
    Works on Hugging Face LlamaForCausalLM (and structural equivalents
    exposing .model.layers[i].mlp and .config.hidden_size).
    """
    config = config or GinobiliConfig()
    core = model.model if hasattr(model, "model") else model
    hidden = model.config.hidden_size
    for layer in core.layers:
        bank = ExpertBank(hidden, config)
        layer.mlp = GraftedMLP(layer.mlp, bank)
    for name, p in model.named_parameters():
        p.requires_grad = ".bank." in name
    return model


def iter_banks(model: nn.Module):
    """Yield (layer_index, ExpertBank) for every grafted layer, in order."""
    core = model.model if hasattr(model, "model") else model
    for i, layer in enumerate(core.layers):
        if isinstance(layer.mlp, GraftedMLP):
            yield i, layer.mlp.bank


def set_top_k(model: nn.Module, k: int | None) -> None:
    """Set the trainer's per-step k override on every bank (None clears)."""
    for _, bank in iter_banks(model):
        bank.force_top_k = k


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
