from dataclasses import dataclass


@dataclass
class GinobiliConfig:
    """Ginobili v1 graft configuration. Defaults are the locked spec."""

    num_experts: int = 12
    top_k: int = 6
    expert_intermediate: int = 512
    full_court_k: int = 12
    # Adaptive widening: if the selected top-k probability mass for a token
    # falls below this threshold, widen k for that token (up to full court).
    widen_threshold: float = 0.55
    # Routing supervision: extra probability mass placed on corpus-tagged
    # experts when building the soft target distribution (0 = no supervision,
    # 1 = hard assignment). Spec: encouragement, not dictation.
    supervision_boost: float = 0.35
    load_balance_weight: float = 0.01
