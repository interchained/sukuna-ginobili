"""Sukuna-Ginobili: grafted micro-expert MoE on a frozen Llama backbone.

Ginobili v1 (architecture lock, 2026-11-26):
- Frozen Llama-3.1-8B-Instruct backbone.
- 12 micro experts per layer (384 modules), compact SwiGLU, intermediate 512.
- Additive expert branch in parallel with the frozen FFN.
- Learned router, top-k per token (k configurable, default 6).
- Expert output (down) projections zero-initialized: the grafted model is
  exactly the stock backbone before training.
"""
from .config import GinobiliConfig
from .experts import MicroExpert, ExpertBank, routing_supervision_loss, load_balance_loss
from .graft import GraftedMLP, graft_model, count_parameters

__all__ = [
    "GinobiliConfig",
    "MicroExpert",
    "ExpertBank",
    "routing_supervision_loss",
    "load_balance_loss",
    "GraftedMLP",
    "graft_model",
    "count_parameters",
]
