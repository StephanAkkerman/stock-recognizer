# Auto-label interactive mode — design

**Date:** 2026-06-02
**File touched:** `utils/auto_label.py`

## Problem

`utils/auto_label.py` currently labels Reddit posts in a two-command cycle per
batch: `--print-prompt` to emit a prompt, paste it into an LLM by hand, save the
reply to a file, then `--response-file` to parse and save. Doing many batches
means repeatedly re-running the script and juggling temp files.

We want an **interactive loop**: the script emits a prompt, the user pastes the
LLM's reply back, and the script immediately emits the next prompt — continuing
until all unlabeled posts are consumed or the user quits.

## Approach

Add an `--interactive` flag to the existing `main()`. When set, it dispatches to
a new `run_interactive()` helper instead of the one-shot path. The helper reuses
the existing pure functions (`build_prompt`, `parse_batch_response`,
`parse_response_to_task`) and the output-save logic. The current inline save
block in `main()` is extracted into a shared `save_tasks(tasks, output)` helper
so both paths use it.

Rejected alternative: a separate script — it would duplicate post loading,
dedup, and the save path. One module is the natural home.

## CLI additions

- `--interactive` — run the loop.
- `--batch-size` (default `10`) — posts per prompt round.
- `--prompt-file` (default `data/auto_label/prompt.txt`) — where each round's
  prompt is written. The parent folder is created if missing.

These are additive; the existing one-shot flags (`--print-prompt`,
`--response-file`, `--posts`, `--post-index`) are unchanged.

## Prompt delivery: file, not terminal

A batch prompt (system instructions + 6 few-shot examples + 10 posts) runs
several thousand characters and scrolls off the terminal, making it hard to
select. So each round **writes the prompt to `--prompt-file`** (overwriting the
same path every round) and prints only a short notice with the path and char
count. The user keeps that file open and re-copies it each round.

The **response** stays in-terminal: the user pastes the LLM JSON directly,
terminated by a sentinel. Responses are smaller and user-produced, so no fit
problem.

## Loop, per round

1. Slice the next `batch_size` posts, sequentially from index 0 through the end
   of the deduped post list.
2. Write the prompt (from `build_prompt`) to `--prompt-file`.
3. Print progress + notice, e.g.:
   ```
   Batch 3 — posts 20–29 of 142
   Prompt written to data/auto_label/prompt.txt (4,812 chars). Copy it into your LLM.
   Paste the JSON response below, then type END on its own line (or q to quit):
   ```
4. Read stdin lines until a line equal to `END` (case-insensitive) **or** EOF
   (Ctrl+Z↵ on Windows). If the collected input, stripped, is exactly `q` or
   `quit` (case-insensitive), stop the loop cleanly.
5. Strip code fence, `json.loads`.
   - **On invalid JSON:** print the error + first 200 chars and re-prompt the
     **same** batch (no advance).
6. Parse with `parse_batch_response` for multi-post batches, or the single-post
   `parse_response_to_task` path for a final 1-post batch.
7. Save incrementally via `save_tasks` (same dedupe-by-task-id as the one-shot
   path), then print the saved/dropped/missing summary.
8. Advance to the next batch. When all posts are consumed, print a done summary
   and return.

## Invariants kept

- Task-id scheme: `task_id_offset + post position` — unchanged, so re-running
  overwrites rather than duplicates.
- Output dedup by task id — unchanged.
- Prompt content (`build_prompt`) — unchanged.
- Saving after **every** round means a mid-session `q` keeps all completed
  batches.

## Controls (per user choice)

- **Quit:** `q` / `quit` at the response prompt → clean stop, progress kept.
- **Retry on bad JSON:** re-prompt the same batch on parse failure.
- (No `skip`, no prompt re-print — declined.)

## Testing

- `read_response_lines` / sentinel parsing: returns collected text on `END`,
  on EOF, and recognizes `q`/`quit` (unit test with a fake stdin).
- `save_tasks`: writing then re-saving the same task id overwrites, doesn't
  duplicate (extends existing save behavior coverage).
- Existing tests for `build_prompt` / `parse_batch_response` continue to pass
  unchanged.
