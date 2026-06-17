"""Auto-label scraped Reddit posts via a reusable few-shot prompt + SOTA LLM.

Workflow (manual mode — recommended for the first batch):
  Single post:
    1) `python utils/auto_label.py --post-index 0 --print-prompt > prompt.txt`
    2) Paste prompt.txt into your LLM, save the JSON reply as response.json
    3) `python utils/auto_label.py --post-index 0 --response-file response.json`

  Batch (recommended for long-context models like Gemini):
    1) `python utils/auto_label.py --posts 0-9 --print-prompt > prompt.txt`
    2) Paste prompt.txt into the LLM, save the JSON reply as response.json
    3) `python utils/auto_label.py --posts 0-9 --response-file response.json`

  The batch output shape is `{"results": [{"index": N, "entities": [...]}, ...]}`
  with one entry per input. Re-running with the same --posts overwrites those
  task IDs rather than duplicating them.

  Interactive mode (loop through everything without juggling files):
    `python utils/auto_label.py --interactive [--batch-size 10]`
    `python utils/auto_label.py --interactive --batch-chars 40000`
    Each round writes the prompt to --prompt-file (default
    data/auto_label/prompt.txt); copy it into your LLM, paste the JSON reply
    back in the terminal, then type END on its own line (or `q` to quit).
    Progress saves after every batch, so quitting mid-run keeps completed work.
    --batch-chars groups posts by total character count instead of post count,
    so a batch of short posts and a batch of long posts consume similar context.

Once the prompt is dialled in (the LLM's outputs match what you'd label by hand
on ~5-10 spot checks), this same module can be called from a thin API wrapper
to scale up — `build_prompt()` and `parse_response_to_task()` are pure functions.

Design choices worth knowing:
  - The LLM returns entity *text* + label, not character offsets. We locate
    offsets ourselves with `re.finditer` — LLMs are notoriously bad at offsets,
    and re-finding text is robust to whitespace/quoting changes.
  - Output lands in `data/preds/`, not `data/labeled/`. Pre-labels need review
    before becoming training data, matching the existing pipeline convention.
  - Posts are deduped against existing labeled+test so the LLM never wastes
    effort on something already annotated.
"""

import argparse
import difflib
import glob
import hashlib
import json
import os
import re
import sys
import textwrap
import uuid

import pandas as pd
from rich.console import Console

console = Console()


# Hand-picked from data/labeled — chosen to cover the patterns v4-v12 stumbled on:
#   1) bare ALL-CAPS ticker → ticker
#   2) lowercase informal shorthand (gme) → ticker; slang stays unlabeled
#   3) cashtag + multi-word company + all-caps ticker (RH: IS the ticker symbol)
#   4) company name with its ticker in parens — both labeled as separate entities
#   5) ALL-CAPS abbreviation where ticker differs (NVIDIA → NVDA) → company
#   6) multi-word company headline + $price-NOT-a-ticker
#   7) all-negative post — slang/finance terms that look entity-shaped but aren't
#
# Edit / extend this list if the LLM starts missing a specific pattern in practice.
FEW_SHOT_EXAMPLES = [
    {
        "input": "I care about BBBY.",
        "output": {"entities": [{"text": "BBBY", "label": "ticker"}]},
    },
    {
        "input": "just loaded up on gme again lmfao",
        "output": {"entities": [{"text": "gme", "label": "ticker"}]},
    },
    {
        "input": "Cash app pulling a RH by not letting people purchase $AMC.What is going on???",
        "output": {
            "entities": [
                {"text": "Cash app", "label": "company"},
                {"text": "RH", "label": "ticker"},
                {"text": "$AMC", "label": "ticker"},
            ]
        },
    },
    {
        "input": "Shorted Nebius (NBIS) by 730,000 today, checkin tomorrow after earnings.",
        "output": {
            "entities": [
                {"text": "Nebius", "label": "company"},
                {"text": "NBIS", "label": "ticker"},
            ]
        },
    },
    {
        "input": "NVIDIA just crushed earnings again, NVDA up 8% after hours.",
        "output": {
            "entities": [
                {"text": "NVIDIA", "label": "company"},
                {"text": "NVDA", "label": "ticker"},
            ]
        },
    },
    {
        "input": "Paramount makes $108.4 billion hostile bid for Warner Bros Discovery",
        "output": {
            "entities": [
                {"text": "Paramount", "label": "company"},
                {"text": "Warner Bros Discovery", "label": "company"},
            ]
        },
    },
    {
        "input": "Anybody else gonna YOLO into puts? JPOW about to crash this market. NFA",
        "output": {"entities": []},
    },
]


BATCH_INSTRUCTIONS = textwrap.dedent("""\
    BATCH MODE: You will receive multiple posts labeled "Input 1:", "Input 2:", etc.
    Return ONE JSON object with this structure (no commentary, no code fence):

    {"results": [
      {"index": 1, "entities": [{"text": "...", "label": "ticker" | "company"}]},
      {"index": 2, "entities": [...]},
      ...
    ]}

    The "index" field MUST match the corresponding "Input N:" number. Include an
    entry for EVERY input, even when entities is empty (use "entities": []).
""")


SYSTEM_INSTRUCTIONS = textwrap.dedent("""\
    You are an expert at extracting stock tickers and company names from Reddit posts about finance.

    TASK: identify every ticker symbol and company name in the input text.

    LABELING RULE — the form of the text in the post determines the label, not the
    author's intent. Apply these rules in order:

    1. CASHTAG ($ directly attached, no space) → always "ticker", no exceptions.
       $AMC, $TSLA, $gme, $EUV, $DRAM → ticker even if the word is also a
       technology term or written in lowercase.
       "$ AUG" (space between $ and word) is NOT a cashtag — do not label it.

    2. ALL-CAPS and the text IS the actual ticker symbol → "ticker".
       AMC, META, SOFI, SNAP, NVDA, SPY, QQQ → ticker.
       ETFs (SPY, QQQ, VOO, IWM) and crypto (BTC, ETH) follow the same rule.

    3. ALL-CAPS but the ticker symbol differs → "company".
       NVIDIA (ticker is NVDA), TSMC (ticker is TSM), APPLE (ticker is AAPL) → company.
       The all-caps abbreviation is the company name, not the tradeable symbol.

    4. Written / mixed-case name → "company".
       Meta, Nvidia, Micron, AMC Theatres, Goldman Sachs, Warner Bros Discovery → company.

    5. Informal lowercase ticker (Reddit shorthand) → "ticker".
       gme, amc, tsla, spy — lowercase versions of ticker symbols → ticker.

    DO NOT LABEL any of the following — they cannot be traded and must be skipped:
    - Internet slang / generics: BUY, SELL, HOLD, DUMP, PUMP, YOLO, DD,
      FUD, FOMO, ATH, ATL, NFA, IMO, TLDR, LMFAO, AMA, OP, MOD.
    - People and macro: JPOW, POWELL, FED, IRS, CEO, CFO, BULL, BEAR,
      MOON, EARNINGS, GAINS, LOSS, PUTS, CALLS, STOCK, MARKET, OPTIONS.
    - Government / regulatory bodies: SEC, CSRC, NASA, FINRA, NDAA, MEXT, METI.
    - Financial metric acronyms: PDT (pattern day trader), EV (enterprise value),
      SGA, RSU, RSUS, PT (price target), ATM, IV, IPO, EBITDA, FCF, NAV.
    - Memory / chip technology terms (without $ prefix): DRAM, HBM, EUV, NAND, GPU.
    - Month abbreviations used in options notation: JAN, FEB, MAR, APR, MAY, JUN,
      JUL, AUG, SEP, OCT, NOV, DEC — e.g. "AUG 19 calls" or "$ AUG 19" means
      August expiry, not a ticker. (With $ directly attached, $AUG would be a ticker.)
    - News outlets and private companies with no public stock:
      CNBC, MSNBC, WSJ, FT, BLOOMBERG, HBO.
    - Dollar amounts: $100, $5.50, $3T — only $-prefixed *letter* sequences are tickers.
    - Index names spelled out: "Dow Jones", "S&P 500", "Nasdaq" (the index itself).

    EDGE CASES:
    - The same company can appear twice with different labels in one post:
      "Meta" → company AND "META" → ticker if both are present.
    - Label EVERY occurrence. If $AAPL appears 5 times, return 5 entries.
    - Preserve exact casing and punctuation from the source — do not normalize.

    OUTPUT FORMAT — return ONLY a JSON object, no commentary, no code fence:
    {"entities": [{"text": "<exact substring from input>", "label": "ticker" | "company"}]}

    Empty entities array (`{"entities": []}`) is correct for posts with no entities.
""")


def build_prompt(texts, examples=None):
    """Assemble: instructions + few-shot examples + target input(s).

    `texts` may be a single string or a list of strings. With one input the
    output format is `{"entities": [...]}`. With 2+ inputs we switch to batch
    mode and ask for `{"results": [{"index": N, "entities": [...]}, ...]}`.
    """
    if isinstance(texts, str):
        texts = [texts]
    is_batch = len(texts) > 1

    examples = examples if examples is not None else FEW_SHOT_EXAMPLES
    parts = [SYSTEM_INSTRUCTIONS]
    if is_batch:
        parts.append(BATCH_INSTRUCTIONS)
    parts.append("EXAMPLES (single-input format, showing what to label):")
    for i, ex in enumerate(examples, 1):
        parts.append(f"\nExample {i}")
        parts.append(f"Input: {ex['input']}")
        parts.append(f"Output: {json.dumps(ex['output'], ensure_ascii=False)}")

    if is_batch:
        parts.append(
            f"\nNow label the following {len(texts)} inputs. "
            "Return ONE JSON object with a 'results' array as described above."
        )
        for i, text in enumerate(texts, 1):
            parts.append(f"\nInput {i}: {text}")
    else:
        parts.append("\nNow label this input. Return ONLY the JSON object.")
        parts.append(f"\nInput: {texts[0]}")
    parts.append("\nOutput:")
    return "\n".join(parts)


def _region_id():
    """Generate a short random region id.

    Label Studio needs a stable per-region ``id`` (alongside ``from_name`` /
    ``to_name``) to map a result span onto the labeling config — without it the
    region is silently dropped and nothing renders on import.
    """
    return uuid.uuid4().hex[:10]


def _resolve_overlaps(spans):
    """Keep the longest span among any that overlap; drop the shorter ones.

    LLMs commonly emit nested variants of the same mention — ``ANPA`` and
    ``$ANPA``, or ``Rich`` / ``Rich Sparkle`` / ``Rich Sparkle Limited`` — which
    land at overlapping offsets and would otherwise render as several stacked
    highlights. Greedy longest-first interval selection keeps the widest span
    and discards anything intersecting it. Non-overlapping occurrences elsewhere
    in the text are unaffected.

    `spans` is a list of ``(start, end, text, label)`` tuples; returns the kept
    subset sorted by start offset.
    """
    # Longest first (tiebreak: earlier start) so the widest variant wins.
    ordered = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    kept = []
    for start, end, ent_text, ent_label in ordered:
        if any(start < k_end and k_start < end for k_start, k_end, _, _ in kept):
            continue
        kept.append((start, end, ent_text, ent_label))
    return sorted(kept, key=lambda s: s[0])


def _fuzzy_find(ent_text, text, threshold=0.8):
    """Find close token-window matches for `ent_text` in `text`.

    Used as a fallback when exact matching fails — e.g. when the LLM normalises
    "Nvdias" (as it appears in the post) to the canonical "Nvidia".  Only called
    for entities of 4+ characters to avoid false positives on short strings.

    Slides a window of N whitespace-delimited tokens across `text` (where N is
    the word count of `ent_text`) and computes a case-insensitive
    SequenceMatcher ratio.  Windows whose ratio exceeds `threshold` are returned
    as ``(start, end, matched_text)`` tuples.
    """
    if len(ent_text) < 4:
        return []
    ent_words = ent_text.split()
    n_words = len(ent_words)
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) < n_words:
        return []
    results = []
    for i in range(len(tokens) - n_words + 1):
        window = tokens[i : i + n_words]
        start = window[0].start()
        end = window[-1].end()
        window_text = text[start:end]
        ratio = difflib.SequenceMatcher(
            None, ent_text.lower(), window_text.lower()
        ).ratio()
        if ratio >= threshold:
            results.append((start, end, window_text))
    return results


def parse_response_to_task(text, response_obj, task_id):
    """Convert LLM JSON response to a Label Studio task, finding offsets in `text`.

    Drops entities whose `text` field can't be located in the source — better
    than emitting fabricated offsets.
    """
    dropped = []
    # The LLM is told to "label every occurrence", and re.finditer below ALSO
    # expands to every occurrence — so a term the LLM lists twice would emit
    # duplicate spans at the same offsets. Dedupe on (start, end, label).
    seen = set()
    spans = []
    for ent in response_obj.get("entities", []):
        ent_text = (ent.get("text") or "").strip()
        ent_label = ent.get("label", "")
        if not ent_text or ent_label not in ("ticker", "company"):
            dropped.append((ent_text, ent_label, "invalid"))
            continue
        matches = list(re.finditer(re.escape(ent_text), text))
        if not matches:
            # Fuzzy fallback: handles LLM normalisations like "Nvdias" → "Nvidia"
            fuzzy = _fuzzy_find(ent_text, text)
            if not fuzzy:
                dropped.append((ent_text, ent_label, "not_found"))
                continue
            for start, end, matched_text in fuzzy:
                span_key = (start, end, ent_label)
                if span_key in seen:
                    continue
                seen.add(span_key)
                spans.append((start, end, matched_text, ent_label))
            continue
        for match in matches:
            span_key = (match.start(), match.end(), ent_label)
            if span_key in seen:
                continue
            seen.add(span_key)
            spans.append((match.start(), match.end(), ent_text, ent_label))

    annotation_results = [
        {
            "id": _region_id(),
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": ent_text,
                "labels": [ent_label],
            },
        }
        for start, end, ent_text, ent_label in _resolve_overlaps(spans)
    ]
    return {
        "id": task_id,
        "data": {"text": text},
        "annotations": [
            {
                "was_cancelled": False,
                "result": annotation_results,
            }
        ],
    }, dropped


def _text_hash(text):
    norm = re.sub(r"\s+", " ", text.strip().lower())[:200]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# Auto-generated WSB daily threads ("What Are Your Moves Tomorrow", "Daily
# Discussion Thread", etc.) have no body text — just this stub pointing at the
# new Reddit UI. Dedup misses them because each carries a unique date/URL, but
# they're identical noise: zero entities, zero training value.
_OLD_REDDIT_STUB = "content not supported on old reddit"


def _is_content_stub(body):
    """True when `body` is just the "not supported on old Reddit" stub.

    Strips markdown links and bare URLs first, then checks whether anything
    beyond the boilerplate sentence remains. A post with real text *and* the
    stub (none exist today, but to be safe) is kept.
    """
    if _OLD_REDDIT_STUB not in body.lower():
        return False
    stripped = re.sub(r"\[[^\]]*\]\([^)]*\)", "", body)  # markdown links
    stripped = re.sub(r"https?://\S+", "", stripped)  # bare URLs
    stripped = re.sub(
        r"(?i)this post contains content not supported on old reddit\.?", "", stripped
    )
    return re.sub(r"\s+", " ", stripped).strip() == ""


def _load_known_hashes(folders):
    hashes = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fp in glob.glob(os.path.join(folder, "*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            for task in data:
                text = (
                    task.get("data", {}).get("text") if isinstance(task, dict) else None
                )
                if text:
                    hashes.add(_text_hash(text))
    return hashes


def load_unlabeled_posts(csv_path, dedup_text_folders=None):
    """Read CSV, combine title+text, dedupe against labeled+test+output corpus."""
    df = pd.read_csv(csv_path)

    known = _load_known_hashes(
        dedup_text_folders or ["data/labeled", "data/test", "data/preds"]
    )

    posts = []
    for _, row in df.iterrows():
        body = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip()
        if body in ("", "nan"):
            continue
        if _is_content_stub(body):
            continue
        combined = (title + "\n\n" + body).strip() if title and title != "nan" else body
        if _text_hash(combined) in known:
            continue
        posts.append(
            {
                "reddit_id": str(row.get("id")),
                "text": combined,
            }
        )
    return posts


def _strip_code_fence(s):
    """LLMs often wrap JSON in ```json ... ``` — peel that off before parsing."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


def parse_post_spec(spec):
    """Parse '0-9', '0,3,5', or '7' into a sorted unique list of indices."""
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def read_until_sentinel(lines):
    """Collect lines until an ``END`` line (case-insensitive) or EOF.

    `lines` is any iterator of strings (e.g. ``sys.stdin``). Returns a
    ``(kind, text)`` tuple. `kind` is ``"quit"`` when a sole ``q``/``quit`` is
    entered before any content (JSON always starts with ``{``/``[``, so this is
    unambiguous), otherwise ``"submit"`` with the newline-joined collected text.
    """
    collected = []
    for line in lines:
        stripped = line.rstrip("\n")
        flat = stripped.strip()
        if flat.upper() == "END":
            break
        if flat.lower() in ("q", "quit") and not any(c.strip() for c in collected):
            return "quit", ""
        collected.append(stripped)
    return "submit", "\n".join(collected)


def save_tasks(tasks, output):
    """Append `tasks` to the JSON array at `output`, overwriting by task id.

    Creates the parent directory if needed. Tasks sharing an id with an
    existing entry replace it (so re-labeling a post overwrites rather than
    duplicates), matching the one-shot ``--response-file`` behavior.
    """
    existing = []
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as f:
            existing = json.load(f)
    new_ids = {t["id"] for t in tasks}
    existing = [t for t in existing if t.get("id") not in new_ids]
    existing.extend(tasks)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def parse_batch_response(texts, response_obj, task_id_offset, post_indices):
    """Convert a batch `{"results": [...]}` response into Label Studio tasks.

    `texts` is the list of input strings (1-indexed by the LLM's "index" field).
    `post_indices` is the corresponding list of original CSV post indices, used
    to assign stable task IDs (so re-running with the same posts overwrites
    rather than duplicates).
    Returns (list_of_tasks, list_of_(input_idx, dropped_entries)).
    """
    results_by_idx = {}
    for r in response_obj.get("results", []):
        idx = r.get("index")
        if isinstance(idx, int):
            results_by_idx[idx] = r

    tasks = []
    all_dropped = []
    missing = []
    for input_idx, text in enumerate(texts, 1):
        result = results_by_idx.get(input_idx)
        if result is None:
            missing.append(input_idx)
            continue
        post_idx = post_indices[input_idx - 1]
        task, dropped = parse_response_to_task(text, result, task_id_offset + post_idx)
        tasks.append(task)
        if dropped:
            all_dropped.append((input_idx, dropped))
    return tasks, all_dropped, missing


def _print_save_summary(tasks, all_dropped, missing, output):
    """Report what was saved, plus any dropped/missing entities."""
    n_ents = sum(len(t["annotations"][0]["result"]) for t in tasks)
    console.print(
        f"[green]Saved {len(tasks)} task(s) ({n_ents} entity spans total) "
        f"to {output}[/green]"
    )
    if missing:
        console.print(
            f"[yellow]Missing results for input index(es): {missing} "
            f"— LLM didn't return entries for these.[/yellow]"
        )
    if all_dropped:
        console.print("[yellow]Dropped entities (not found in source text):[/yellow]")
        for inp_i, dropped in all_dropped:
            for txt, lab, reason in dropped:
                console.print(f"  - input {inp_i}: {reason}: {txt!r} ({lab})")


def _response_to_tasks(texts, response_obj, task_id_offset, post_indices):
    """Dispatch to batch or single parsing based on input count."""
    if len(texts) > 1:
        return parse_batch_response(texts, response_obj, task_id_offset, post_indices)
    task, dropped = parse_response_to_task(
        texts[0], response_obj, task_id_offset + post_indices[0]
    )
    return [task], ([(1, dropped)] if dropped else []), []


def _build_char_batches(posts, max_chars):
    """Group posts into contiguous (start, end) slices where total text ≤ max_chars.

    A single post that exceeds max_chars is emitted as its own one-post batch
    rather than being silently skipped.
    """
    batches = []
    start = 0
    while start < len(posts):
        total = 0
        end = start
        while end < len(posts):
            n = len(posts[end]["text"])
            if end > start and total + n > max_chars:
                break
            total += n
            end += 1
        batches.append((start, end))
        start = end
    return batches


def run_interactive(posts, args, line_source=None):
    """Loop over all `posts` in batches, prompting + reading replies in-terminal.

    Each round writes the batch prompt to ``args.prompt_file`` (overwriting it),
    then reads the pasted LLM JSON from `line_source` (default ``sys.stdin``)
    until an ``END`` line / EOF, or ``q`` to quit. Saves after every batch so a
    mid-session quit keeps completed work. Bad JSON re-prompts the same batch.
    """
    line_source = sys.stdin if line_source is None else line_source
    total = len(posts)

    batch_chars = getattr(args, "batch_chars", None)
    if batch_chars:
        batch_slices = _build_char_batches(posts, batch_chars)
    else:
        b = args.batch_size
        batch_slices = [(i, min(i + b, total)) for i in range(0, total, b)]
    n_batches = len(batch_slices)

    batch_idx = 0
    while batch_idx < n_batches:
        start, end = batch_slices[batch_idx]
        post_indices = list(range(start, end))
        texts = [posts[i]["text"] for i in post_indices]
        char_count = sum(len(t) for t in texts)

        prompt = build_prompt(texts)
        parent = os.path.dirname(args.prompt_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        console.print(
            f"\n[bold cyan]Batch {batch_idx + 1}/{n_batches} — "
            f"posts {start}–{end - 1} ({len(texts)} posts, {char_count:,} chars)[/bold cyan]"
        )
        console.print(
            f"Prompt written to [bold]{args.prompt_file}[/bold] "
            f"({len(prompt):,} chars). Copy it into your LLM."
        )
        console.print(
            "[dim]Paste the JSON response below, then type END on its own line "
            "(or q to quit):[/dim]"
        )

        kind, raw = read_until_sentinel(line_source)
        if kind == "quit":
            console.print("[yellow]Quit — progress saved.[/yellow]")
            return
        raw = _strip_code_fence(raw)
        if not raw.strip():
            console.print("[yellow]Empty response — stopping. Progress saved.[/yellow]")
            return

        try:
            response_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Invalid JSON: {exc}[/red]")
            console.print(f"[dim]First 200 chars: {raw[:200]}[/dim]")
            console.print("[yellow]Re-paste the response for this batch.[/yellow]")
            continue  # retry the same batch — batch_idx unchanged

        tasks, all_dropped, missing = _response_to_tasks(
            texts, response_obj, args.task_id_offset, post_indices
        )
        save_tasks(tasks, args.output)
        _print_save_summary(tasks, all_dropped, missing, args.output)
        batch_idx += 1

    console.print(f"\n[bold green]Done — all {total} posts processed.[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Build / parse auto-label prompts for Reddit posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--csv",
        default="data/wallstreetbets_posts.csv",
        help="Scraped posts CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--post-index", type=int, help="Single-post shortcut, equivalent to --posts N."
    )
    parser.add_argument(
        "--posts",
        default=None,
        help="Posts to label. '0-9' for a range, '0,3,5' for a list, "
        "or a single index. Defaults to '0'.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Write the full prompt to stdout (suitable for pasting into an LLM).",
    )
    parser.add_argument(
        "--response-file",
        help="Path to an LLM JSON response. Parses + appends to --output.",
    )
    parser.add_argument(
        "--task-id-offset",
        type=int,
        default=8_000_000,
        help="Starting ID for auto-labeled tasks (avoids labeled.json collisions).",
    )
    parser.add_argument(
        "--output",
        default="data/preds/auto_labeled.json",
        help="Where to append parsed Label Studio tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Loop over all unlabeled posts: emit a prompt, paste the "
        "LLM reply, repeat. Ignores --posts/--post-index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Posts per prompt in --interactive mode (default: %(default)s). "
        "Ignored when --batch-chars is set.",
    )
    parser.add_argument(
        "--batch-chars",
        type=int,
        default=None,
        help="Max total post characters per --interactive batch. "
        "Overrides --batch-size. Posts are grouped until adding "
        "the next post would exceed this limit.",
    )
    parser.add_argument(
        "--prompt-file",
        default="data/auto_label/prompt.txt",
        help="Where --interactive writes each round's prompt "
        "(default: %(default)s).",
    )
    args = parser.parse_args()

    posts = load_unlabeled_posts(args.csv)
    if not posts:
        console.print(f"[red]No unlabeled posts in {args.csv} after dedup.[/red]")
        sys.exit(1)

    if args.interactive:
        run_interactive(posts, args)
        return

    # Resolve the post-index spec into a concrete list of CSV positions.
    if args.posts is not None:
        post_indices = parse_post_spec(args.posts)
    elif args.post_index is not None:
        post_indices = [args.post_index]
    else:
        post_indices = [0]

    for idx in post_indices:
        if idx < 0 or idx >= len(posts):
            console.print(
                f"[red]post index {idx} out of range (0..{len(posts)-1}).[/red]"
            )
            sys.exit(1)

    targets = [posts[i] for i in post_indices]
    texts = [t["text"] for t in targets]
    is_batch = len(targets) > 1

    if args.response_file:
        with open(args.response_file, "r", encoding="utf-8") as f:
            raw = _strip_code_fence(f.read())
        try:
            response_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Response file is not valid JSON: {exc}[/red]")
            console.print(f"[dim]First 200 chars: {raw[:200]}[/dim]")
            sys.exit(1)

        tasks, all_dropped, missing = _response_to_tasks(
            texts, response_obj, args.task_id_offset, post_indices
        )
        save_tasks(tasks, args.output)
        _print_save_summary(tasks, all_dropped, missing, args.output)
        return

    prompt = build_prompt(texts)

    if args.print_prompt:
        sys.stdout.write(prompt)
        sys.stdout.write("\n")
        return

    range_desc = (
        f"posts {post_indices[0]}-{post_indices[-1]} ({len(targets)} total)"
        if is_batch
        else f"post {post_indices[0]}"
    )
    console.print(
        f"\n[bold cyan]{range_desc}[/bold cyan] "
        f"of {len(posts)-1} available "
        f"({sum(len(t) for t in texts)} chars total input)"
    )
    if not is_batch:
        console.print("[dim]--- POST PREVIEW ---[/dim]")
        preview = texts[0][:400].replace("\n", " ")
        console.print(preview + ("..." if len(texts[0]) > 400 else ""))
    console.print(
        f"\n[dim]--- PROMPT ({len(prompt)} chars) — "
        f"use --print-prompt to write raw to stdout ---[/dim]\n"
    )
    print(prompt)


if __name__ == "__main__":
    main()
