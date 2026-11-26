"""Ginobili training step: LM loss + routing supervision + load balance.

Per step:
- k is sampled from `k_choices` and set on every bank, so the model is
  trained across formations (the full-court flag is a practiced setting,
  not an inference-time improvisation).
- LM loss is the backbone's own causal loss over assistant-masked labels.
- Routing supervision (soft, from corpus expert tags) and the load
  balance term are averaged over grafted layers.
Only bank parameters receive gradients; the caller builds the optimizer
over trainable parameters only.
"""
import random

import torch

from .config import GinobiliConfig
from .experts import load_balance_loss, routing_supervision_loss
from .graft import GraftedMLP, iter_banks, set_top_k


def _routing_terms(model, expert_targets, config: GinobiliConfig):
    sup_terms, lb_terms = [], []
    core = model.model if hasattr(model, "model") else model
    for layer in core.layers:
        if not isinstance(layer.mlp, GraftedMLP):
            continue
        routing = layer.mlp.last_routing
        if routing is None:
            continue
        probs = routing["probs"]
        b = expert_targets.shape[0]
        targets = expert_targets.to(probs.dtype).reshape(b, 1, -1).expand_as(probs)
        sup_terms.append(routing_supervision_loss(probs, targets, config.supervision_boost))
        lb_terms.append(load_balance_loss(probs, routing["indices"], routing["k"]))
    if not sup_terms:
        return None, None
    return torch.stack(sup_terms).mean(), torch.stack(lb_terms).mean()


def train_steps(model, batches, config: GinobiliConfig | None = None,
                lr: float = 1e-3, k_choices=(4, 6, 8, 12), seed: int = 0,
                supervision_weight: float = 1.0) -> list[dict]:
    """Run one pass over `batches` (already-collated dicts). Returns logs."""
    config = config or GinobiliConfig()
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    logs = []
    model.train()
    for step, batch in enumerate(batches):
        k = rng.choice(k_choices)
        set_top_k(model, k)
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        sup, lb = _routing_terms(model, batch["expert_targets"], config)
        loss = out.loss
        if sup is not None:
            loss = loss + supervision_weight * sup + config.load_balance_weight * lb
        opt.zero_grad()
        loss.backward()
        opt.step()
        logs.append({
            "step": step,
            "k": k,
            "loss": float(loss.detach()),
            "lm_loss": float(out.loss.detach()),
            "routing_supervision": None if sup is None else float(sup.detach()),
            "load_balance": None if lb is None else float(lb.detach()),
        })
    set_top_k(model, None)
    return logs


def expert_weight_movement(model) -> dict:
    """Per named expert: total |down_proj weight| summed over layers.

    Zero for a sleeping expert; positive once it has started learning.
    This is the Gate 1 awakening readout.
    """
    from .data import EXPERTS
    movement = {name: 0.0 for name in EXPERTS}
    for _, bank in iter_banks(model):
        for i, expert in enumerate(bank.experts):
            movement[EXPERTS[i]] += float(expert.down_proj.weight.detach().abs().sum())
    return movement
