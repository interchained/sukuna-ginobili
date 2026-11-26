"""Gate 0 (identity preservation) and Gate 1 (awakening) proofs, in miniature.

Runs on a tiny random Llama so the full harness executes on CPU in seconds.
The same checks run against the real 8B on the B200 before training.
"""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ginobili import (GinobiliConfig, MicroExpert, count_parameters,
                      graft_model, routing_supervision_loss)


def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256)
    return LlamaForCausalLM(cfg)


def test_expert_param_count_matches_spec():
    e = MicroExpert(4096, 512)
    assert sum(p.numel() for p in e.parameters()) == 6_291_456


def test_gate0_identity_at_init():
    model = tiny_model()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        before = model(ids).logits
    graft_model(model, GinobiliConfig(expert_intermediate=32))
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(before, after, atol=1e-6), "graft must be invisible at init"


def test_only_banks_train():
    model = tiny_model()
    graft_model(model, GinobiliConfig(expert_intermediate=32))
    counts = count_parameters(model)
    # 2 layers x (12 experts x 3 x 64 x 32 + router 64 x 12)
    per_layer = 12 * 3 * 64 * 32 + 64 * 12
    assert counts["trainable"] == 2 * per_layer
    assert counts["frozen"] > 0


def test_gate0_no_frozen_gradients_and_zero_init_dynamics():
    model = tiny_model()
    graft_model(model, GinobiliConfig(expert_intermediate=32))
    ids = torch.randint(0, 256, (2, 16))
    loss = model(ids, labels=ids).loss
    loss.backward()
    for name, p in model.named_parameters():
        if ".bank." not in name:
            assert p.grad is None, f"frozen param received gradient: {name}"
    banks = [m for m in model.modules() if m.__class__.__name__ == "GraftedMLP"]
    down_grads, gate_grads, router_grads = [], [], []
    for g in banks:
        for e in g.bank.experts:
            down_grads.append(e.down_proj.weight.grad.abs().sum().item())
            gate_grads.append(e.gate_proj.weight.grad.abs().sum().item())
        router_grads.append(g.bank.router.weight.grad.abs().sum().item())
    # Zero-init dynamics: output projections learn first; input projections
    # and the router receive exactly zero gradient through the LM loss.
    assert sum(down_grads) > 0
    assert sum(gate_grads) == 0.0
    assert sum(router_grads) == 0.0


def test_router_trains_via_supervision_at_init():
    model = tiny_model()
    graft_model(model, GinobiliConfig(expert_intermediate=32))
    layer = model.model.layers[0].mlp
    z = torch.randn(2, 5, 64)
    _, routing = layer.bank(z)
    target = torch.zeros(2, 5, 12)
    target[..., 0] = 1.0  # pretend every token is tagged Abacus
    loss = routing_supervision_loss(routing["probs"], target, boost=0.35)
    loss.backward()
    assert layer.bank.router.weight.grad.abs().sum().item() > 0


def test_topk_configurable_and_full_court():
    model = tiny_model()
    graft_model(model, GinobiliConfig(expert_intermediate=32))
    bank = model.model.layers[0].mlp.bank
    z = torch.randn(2, 5, 64)
    out, routing = bank(z, full_court=True)
    assert out.shape == z.shape
    assert int(routing["k"].min()) == 12
    out2, _ = bank(z, top_k=2)
    assert out2.shape == z.shape
