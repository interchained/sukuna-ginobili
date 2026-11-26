# Ginobili corpus plan

Target: 500k records per teacher, three teachers (Muse, Oracle,
Grok), 1.5M raw. First run floor: 100k combined - Mark, 2026-11-26;
with Grok added the first run is 150k (50k per teacher).
Grading trims an expected 20-30% before training.

Grok's seat (Mark, 2026-11-26): weighted toward Hearth, Atlas,
Prism, Polyglot; lighter on Abacus, Vector, Syntax. Grok does not
grade in round one - Muse and the Oracle split grading Grok's
records until its pass rate earns a grading seat. The teacher field
records the source honestly for later per-teacher ablation.

## Canonical record shape

The standard shape in the spec governs. Tool calls use the key form,
exactly as the trainer renders them:

    {"role": "assistant", "tool_call": {"name": "calculator",
     "arguments": {"expression": "2450 * 0.17"}}}
    {"role": "tool", "name": "calculator", "content": "416.5"}

Standard tools: calculator, code_runner, database_query, web_search,
file_read, calendar. Every record ends with a non-empty assistant
message. Grade is null at generation time.

Validate everything with tools/validate_corpus.py before it counts.

## Mix targets (per teacher, at 50k)

- Type: chat 55%, tool_trajectory 25%, structured 20%
- Difficulty: 1: 10%, 2: 25%, 3: 35%, 4: 20%, 5: 10%
- Language: en 70%, es 6%, pt 4%, fr 4%, de 3%, zh 3%, ja 2%,
  hi 2%, ar 2%, other 4%
- Experts: every expert at least 3,000 primary tags; Hearth and
  Operator carry the largest shares (conversation and tool use are
  the product's face)

## Batch protocol

- 100 records per file, in corpus/claude-batches/ (Muse) with ids
  continuing the teacher sequence (Muse seed = 000001-000016, so
  generation starts at claude-000017; oracle likewise from 000017).
- A batch counts only when validate_corpus.py passes it clean.
- Progress is the count of validated records on disk, nothing else.
