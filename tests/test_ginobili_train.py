"""Gate 1 (expert awakening) in miniature: train the tiny grafted model on
real standard-shape records and verify learning actually happens."""
import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM

from ginobili import (GinobiliConfig, GinobiliDataset, collate,
                      expert_weight_movement, graft_model, train_steps)

RECORDS = [
    {"id": "t-1", "teacher": "Muse", "experts": ["Abacus"], "domain": "mathematics",
     "language": "en", "type": "chat", "difficulty": 2, "grade": None,
     "messages": [{"role": "user", "content": "What is 17 + 25?"},
                  {"role": "assistant", "content": "17 + 25 = 42."}]},
    {"id": "t-2", "teacher": "Muse", "experts": ["Hearth"], "domain": "conversation",
     "language": "en", "type": "chat", "difficulty": 1, "grade": None,
     "messages": [{"role": "user", "content": "Hello, how are you?"},
                  {"role": "assistant", "content": "Hello! I am well, thank you for asking."}]},
    {"id": "t-3", "teacher": "Muse", "experts": ["Operator", "Abacus"],
     "domain": "agentic", "language": "en", "type": "tool_trajectory",
     "difficulty": 3, "grade": None,
     "messages": [{"role": "user", "content": "What is 6 * 7?"},
                  {"role": "assistant", "tool_call": {"name": "calculator",
                                                      "arguments": {"expression": "6*7"}}},
                  {"role": "tool", "name": "calculator", "content": "42"},
                  {"role": "assistant", "content": "6 * 7 = 42."}]},
]


def byte_tokenize(text: str):
    return list(text.encode("utf-8"))


def tiny_grafted():
    torch.manual_seed(1)
    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=256)
    model = LlamaForCausalLM(cfg)
    graft_model(model, GinobiliConfig(expert_intermediate=32))
    return model


def test_assistant_span_masking():
    ds = GinobiliDataset(RECORDS, byte_tokenize)
    ex = ds[0]
    labeled = [t for t in ex["labels"] if t != -100]
    text = bytes(labeled).decode("utf-8")
    assert "42" in text and "17 + 25" in text  # assistant span carries loss
    user_only = bytes(t for t in ex["input_ids"]).decode("utf-8")
    assert "What is" in user_only  # user text present in inputs
    tool_ex = ds[2]
    tool_labels = bytes(t for t in tool_ex["labels"] if t != -100).decode("utf-8")
    assert "tool_call" in tool_labels  # tool calls are learned
    assert tool_ex["expert_target"][9] == 1.0  # Operator tagged


def test_gate1_awakening_in_miniature():
    model = tiny_grafted()
    ds = GinobiliDataset(RECORDS, byte_tokenize)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate)
    batches = list(loader) * 4  # 4 steps over the same 3 records
    before = expert_weight_movement(model)
    assert all(v == 0.0 for v in before.values())
    logs = train_steps(model, batches, lr=5e-3, k_choices=(12,), seed=0)
    after = expert_weight_movement(model)
    # Full-court steps route every token to every expert: the whole bench wakes.
    assert all(v > 0.0 for v in after.values()), after
    assert logs[-1]["loss"] < logs[0]["loss"] + 1.0  # no divergence
    assert logs[0]["routing_supervision"] is not None


def test_k_sampling_runs():
    model = tiny_grafted()
    ds = GinobiliDataset(RECORDS, byte_tokenize)
    loader = DataLoader(ds, batch_size=3, collate_fn=collate)
    logs = train_steps(model, list(loader) * 3, lr=1e-3,
                       k_choices=(2, 6, 12), seed=3)
    assert {log["k"] for log in logs} <= {2, 6, 12}
    assert all(log["loss"] == log["loss"] for log in logs)  # finite
