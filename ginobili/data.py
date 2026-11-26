"""Corpus loading for the Ginobili standard JSONL shape.

Renders each record's messages into one training sequence with
assistant-span loss masking (inherited from Sukuna-Imagine): loss applies
to assistant text and assistant tool calls, never to user or tool content.

Rendering is segment-wise: each message becomes one segment, tokenized
separately and concatenated. With BPE tokenizers this can differ from
whole-text tokenization at segment boundaries; acceptable for v1 and
identical in spirit to the legacy trainer's packing.
"""
import json

import torch
from torch.utils.data import Dataset

EXPERTS = ["Abacus", "Vector", "Syntax", "Circuit", "Ledger", "Empiric",
           "Polyglot", "Prism", "Hearth", "Operator", "Atlas", "Sentinel"]
EXPERT_INDEX = {name: i for i, name in enumerate(EXPERTS)}

_ROLE_OPEN = {"user": "<|user|>\n", "assistant": "<|assistant|>\n",
              "tool": "<|tool|>\n", "system": "<|system|>\n"}


def load_corpus(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def render_record(record: dict) -> list[tuple[str, bool]]:
    """Return (segment_text, is_assistant) pairs for one record."""
    segments = []
    for msg in record["messages"]:
        if "tool_call" in msg:
            call = msg["tool_call"]
            body = "<tool_call>" + json.dumps(
                {"name": call["name"], "arguments": call["arguments"]},
                ensure_ascii=False) + "</tool_call>"
            segments.append((_ROLE_OPEN["assistant"] + body + "\n", True))
        elif msg["role"] == "tool":
            segments.append((_ROLE_OPEN["tool"] + str(msg["content"]) + "\n", False))
        else:
            role = msg["role"]
            segments.append((_ROLE_OPEN[role] + str(msg["content"]) + "\n",
                             role == "assistant"))
    return segments


def expert_target(record: dict) -> torch.Tensor:
    """Multi-hot routing target over the 12 experts from the record's tags."""
    target = torch.zeros(len(EXPERTS))
    for name in record.get("experts", []):
        if name in EXPERT_INDEX:
            target[EXPERT_INDEX[name]] = 1.0
    return target


class GinobiliDataset(Dataset):
    """Tokenizes rendered records; `tokenize` maps text -> list[int]."""

    def __init__(self, records: list[dict], tokenize, max_len: int = 2048):
        self.examples = []
        for record in records:
            ids, labels = [], []
            for text, is_assistant in render_record(record):
                seg = tokenize(text)
                ids.extend(seg)
                labels.extend(seg if is_assistant else [-100] * len(seg))
            if len(ids) > max_len:
                ids, labels = ids[:max_len], labels[:max_len]
            self.examples.append({
                "input_ids": ids,
                "labels": labels,
                "expert_target": expert_target(record),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch: list[dict], pad_id: int = 0) -> dict:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, targets = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        targets.append(b["expert_target"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "expert_targets": torch.stack(targets),
    }
