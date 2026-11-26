#!/usr/bin/env python3
"""Validate Ginobili corpus JSONL files against the standard record shape.

Usage: validate_corpus.py <file-or-dir> [...]
Exit code 0 when every record conforms. Prints per-file counts and errors.
"""
import json
import sys
from pathlib import Path

EXPERTS = {"Abacus", "Vector", "Syntax", "Circuit", "Ledger", "Empiric",
           "Polyglot", "Prism", "Hearth", "Operator", "Atlas", "Sentinel"}
TYPES = {"chat", "tool_trajectory", "structured"}
TOOLS = {"calculator", "code_runner", "database_query", "web_search",
         "file_read", "calendar"}


def check_record(r):
    errs = []
    for f in ("id", "teacher", "experts", "domain", "language", "type",
              "difficulty", "messages", "grade"):
        if f not in r:
            errs.append(f"missing field {f}")
    if errs:
        return errs
    if r["teacher"] not in ("Muse", "oracle", "grok"):
        errs.append(f"teacher={r['teacher']!r}")
    ex = r["experts"]
    if not (1 <= len(ex) <= 3):
        errs.append(f"{len(ex)} experts tagged")
    for e in ex:
        if e not in EXPERTS:
            errs.append(f"unknown expert {e}")
    if r["type"] not in TYPES:
        errs.append(f"type={r['type']!r}")
    if not (1 <= r["difficulty"] <= 5):
        errs.append(f"difficulty={r['difficulty']}")
    msgs = r["messages"]
    if not msgs:
        errs.append("no messages")
        return errs
    if msgs[0].get("role") != "user":
        errs.append("first message not user")
    if msgs[-1].get("role") != "assistant" or not msgs[-1].get("content"):
        errs.append("must end with a non-empty assistant message")
    if r["type"] == "tool_trajectory":
        calls = [m for m in msgs if isinstance(m.get("tool_call"), dict)]
        if not calls:
            errs.append("tool_trajectory without a tool_call message")
        for m in calls:
            if m.get("role") != "assistant":
                errs.append("tool_call must ride on an assistant message")
            c = m["tool_call"]
            if c.get("name") not in TOOLS or not isinstance(c.get("arguments"), dict):
                errs.append("tool_call must be {name: <standard tool>, arguments: {...}}")
        if not any(m.get("role") == "tool" for m in msgs):
            errs.append("tool_trajectory without tool result")
    else:
        if any("tool_call" in m or m.get("role") == "tool" for m in msgs):
            errs.append(f"{r['type']} record contains tool messages")
    if r["grade"] is not None:
        g = r["grade"]
        if not isinstance(g, dict) or "score" not in g:
            errs.append("grade must be null or {score, ...}")
    return errs


def validate_file(path):
    total, bad = 0, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        try:
            r = json.loads(line)
        except Exception as e:
            bad.append(f"line {total}: bad JSON: {e}")
            continue
        for e in check_record(r):
            bad.append(f"{r.get('id', f'line {total}')}: {e}")
    return total, bad


def main():
    files = []
    for a in sys.argv[1:]:
        p = Path(a)
        files.extend(sorted(p.rglob("*.jsonl")) if p.is_dir() else [p])
    grand, errors = 0, 0
    for f in files:
        n, bad = validate_file(f)
        grand += n
        errors += len(bad)
        print(f"{f}: {n} records, {len(bad)} errors")
        for b in bad[:10]:
            print("   ", b)
    print(f"TOTAL: {grand} records, {errors} errors")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
