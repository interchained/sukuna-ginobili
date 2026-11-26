import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GinobiliConfig


class MicroExpert(nn.Module):
    """One compact SwiGLU expert: E(z) = W_d( SiLU(W_g z) * W_u z ).

    Bias-free projections, matching Llama's MLP. The down projection is
    zero-initialized so the expert contributes exactly nothing until
    training moves it (Gate 0 identity preservation).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(z)) * self.up_proj(z))


class ExpertBank(nn.Module):
    """A per-layer bank of micro experts plus its learned router.

    forward(z) returns (branch_output, routing) where routing carries the
    full router probability distribution (for auxiliary losses) and the
    per-token selected indices.
    """

    def __init__(self, hidden_size: int, config: GinobiliConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            MicroExpert(hidden_size, config.expert_intermediate)
            for _ in range(config.num_experts)
        )
        # Trainer-set override (k-sampling during training); None = use
        # the caller's top_k or the config default.
        self.force_top_k: int | None = None

    def _effective_k(self, probs: torch.Tensor, top_k: int, full_court: bool) -> torch.Tensor:
        """Per-token k: full court forces max; otherwise widen when unsure."""
        cfg = self.config
        n = cfg.num_experts
        if full_court:
            return torch.full(probs.shape[:-1], min(cfg.full_court_k, n),
                              dtype=torch.long, device=probs.device)
        k = torch.full(probs.shape[:-1], min(top_k, n),
                       dtype=torch.long, device=probs.device)
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        base_mass = sorted_probs[..., : min(top_k, n)].sum(dim=-1)
        unsure = base_mass < cfg.widen_threshold
        # Widen in steps up to full court while mass stays below threshold.
        for cand in range(min(top_k, n) + 1, min(cfg.full_court_k, n) + 1):
            widen = unsure & (sorted_probs[..., :cand].sum(dim=-1) < cfg.widen_threshold + 0.25)
            k = torch.where(widen, torch.full_like(k, cand), k)
            unsure = unsure & widen
        return k

    def forward(self, z: torch.Tensor, top_k: int | None = None,
                full_court: bool = False):
        cfg = self.config
        if top_k is None:
            top_k = self.force_top_k
        k_req = cfg.top_k if top_k is None else top_k
        probs = F.softmax(self.router(z), dim=-1)
        ks = self._effective_k(probs, k_req, full_court)
        max_k = int(ks.max().item())
        top_vals, top_idx = probs.topk(max_k, dim=-1)
        flat_z = z.reshape(-1, z.shape[-1])
        flat_out = torch.zeros_like(flat_z)
        flat_idx = top_idx.reshape(-1, max_k)
        flat_vals = top_vals.reshape(-1, max_k)
        flat_k = ks.reshape(-1)
        for e, expert in enumerate(self.experts):
            mask = (flat_idx == e) & (torch.arange(max_k, device=z.device).unsqueeze(0) < flat_k.unsqueeze(1))
            rows, cols = mask.nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            contrib = expert(flat_z[rows]) * flat_vals[rows, cols].unsqueeze(-1)
            flat_out.index_add_(0, rows, contrib)
        out = flat_out.reshape_as(z)
        routing = {"probs": probs, "indices": top_idx, "values": top_vals, "k": ks}
        return out, routing


def routing_supervision_loss(router_probs: torch.Tensor,
                             target_experts: torch.Tensor,
                             boost: float) -> torch.Tensor:
    """Soft routing supervision: encouragement, not dictation.

    target_experts: (..., E) multi-hot of corpus-tagged experts for each
    token (sequence-level tags broadcast to tokens). The target
    distribution is uniform with `boost` extra total mass spread over the
    tagged experts; loss is KL(target || router_probs).
    """
    e = router_probs.shape[-1]
    target = torch.full_like(router_probs, 1.0 / e)
    tagged = target_experts.to(router_probs.dtype)
    counts = tagged.sum(dim=-1, keepdim=True).clamp(min=1.0)
    target = target * (1.0 - boost) + tagged / counts * boost
    return F.kl_div(router_probs.clamp(min=1e-9).log(), target, reduction="batchmean")


def load_balance_loss(router_probs: torch.Tensor,
                      indices: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
    """Switch-style load balancing: E * sum_e( fraction_e * mean_prob_e )."""
    e = router_probs.shape[-1]
    flat_probs = router_probs.reshape(-1, e)
    flat_idx = indices.reshape(-1, indices.shape[-1])
    flat_k = ks.reshape(-1)
    mask = (torch.arange(flat_idx.shape[1], device=indices.device).unsqueeze(0)
            < flat_k.unsqueeze(1))
    one_hot = F.one_hot(flat_idx, num_classes=e).to(router_probs.dtype) * mask.unsqueeze(-1)
    fraction = one_hot.sum(dim=(0, 1)) / mask.sum().clamp(min=1)
    mean_prob = flat_probs.mean(dim=0)
    return e * (fraction * mean_prob).sum()
