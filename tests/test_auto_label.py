import json
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import auto_label  # noqa: E402


def _args(tmp_path, batch_size=2):
    return types.SimpleNamespace(
        batch_size=batch_size,
        task_id_offset=1000,
        output=str(tmp_path / "out.json"),
        prompt_file=str(tmp_path / "auto_label" / "prompt.txt"),
    )


def _task(task_id, text="hello"):
    return {
        "id": task_id,
        "data": {"text": text},
        "annotations": [{"was_cancelled": False, "result": []}],
    }


# --- read_until_sentinel -------------------------------------------------


def test_read_until_sentinel_stops_at_end():
    lines = iter(['{"results": [', '  {"index": 1}', "]}", "END", "ignored"])
    kind, text = auto_label.read_until_sentinel(lines)
    assert kind == "submit"
    assert text == '{"results": [\n  {"index": 1}\n]}'


def test_read_until_sentinel_stops_at_eof():
    lines = iter(['{"entities": []}'])
    kind, text = auto_label.read_until_sentinel(lines)
    assert kind == "submit"
    assert text == '{"entities": []}'


def test_read_until_sentinel_quit_q():
    kind, text = auto_label.read_until_sentinel(iter(["q"]))
    assert kind == "quit"


def test_read_until_sentinel_quit_word_case_insensitive():
    kind, text = auto_label.read_until_sentinel(iter(["Quit"]))
    assert kind == "quit"


def test_read_until_sentinel_q_inside_json_is_not_quit():
    # A 'q' is only a quit command before any real content is collected.
    lines = iter(['{"entities": [{"text": "q", "label": "ticker"}]}', "END"])
    kind, text = auto_label.read_until_sentinel(lines)
    assert kind == "submit"
    assert "label" in text


# --- save_tasks ----------------------------------------------------------


def test_save_tasks_creates_parent_and_writes(tmp_path):
    out = tmp_path / "nested" / "preds.json"
    auto_label.save_tasks([_task(1)], str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [t["id"] for t in data] == [1]


def test_save_tasks_overwrites_same_id(tmp_path):
    out = tmp_path / "preds.json"
    auto_label.save_tasks([_task(5, "first")], str(out))
    auto_label.save_tasks([_task(5, "second")], str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    same_id = [t for t in data if t["id"] == 5]
    assert len(same_id) == 1
    assert same_id[0]["data"]["text"] == "second"


def test_save_tasks_appends_new_ids(tmp_path):
    out = tmp_path / "preds.json"
    auto_label.save_tasks([_task(1)], str(out))
    auto_label.save_tasks([_task(2)], str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert sorted(t["id"] for t in data) == [1, 2]


# --- run_interactive -----------------------------------------------------


def test_run_interactive_processes_all_batches(tmp_path):
    posts = [
        {"text": "I love AAPL"},
        {"text": "nothing here"},
        {"text": "TSLA to the moon"},
    ]
    args = _args(tmp_path, batch_size=2)
    # Batch 1 (posts 0-1, batch format) then batch 2 (post 2, single format).
    stdin = iter(
        [
            '{"results": [{"index": 1, "entities": [{"text": "AAPL", "label": "ticker"}]},'
            ' {"index": 2, "entities": []}]}',
            "END",
            '{"entities": [{"text": "TSLA", "label": "ticker"}]}',
            "END",
        ]
    )
    auto_label.run_interactive(posts, args, line_source=stdin)

    data = json.loads(open(args.output, encoding="utf-8").read())
    # Task ids = offset + post position; all three posts saved.
    assert sorted(t["id"] for t in data) == [1000, 1001, 1002]
    by_id = {t["id"]: t for t in data}
    assert by_id[1000]["annotations"][0]["result"][0]["value"]["text"] == "AAPL"
    assert by_id[1001]["annotations"][0]["result"] == []
    assert by_id[1002]["annotations"][0]["result"][0]["value"]["text"] == "TSLA"
    # The prompt file was written (and its parent dir created).
    assert os.path.exists(args.prompt_file)


def test_run_interactive_quit_keeps_prior_progress(tmp_path):
    posts = [{"text": "AAPL"}, {"text": "TSLA"}, {"text": "MSFT"}]
    args = _args(tmp_path, batch_size=2)
    stdin = iter(
        [
            '{"results": [{"index": 1, "entities": [{"text": "AAPL", "label": "ticker"}]},'
            ' {"index": 2, "entities": []}]}',
            "END",
            "q",  # quit before labeling batch 2
        ]
    )
    auto_label.run_interactive(posts, args, line_source=stdin)

    data = json.loads(open(args.output, encoding="utf-8").read())
    assert sorted(t["id"] for t in data) == [1000, 1001]


def test_parse_response_dedupes_repeated_entities():
    # LLM lists "Paramount" twice; the source text contains it once. Without
    # dedup, re.finditer would emit two identical spans at 0-9.
    text = "Paramount bids for Warner Bros Discovery"
    response = {
        "entities": [
            {"text": "Paramount", "label": "company"},
            {"text": "Paramount", "label": "company"},
            {"text": "Warner Bros Discovery", "label": "company"},
        ]
    }
    task, dropped = auto_label.parse_response_to_task(text, response, 1)
    results = task["annotations"][0]["result"]
    spans = [(r["value"]["start"], r["value"]["end"]) for r in results]
    assert spans == [(0, 9), (19, 40)]


def test_parse_response_keeps_distinct_occurrences():
    # A genuinely repeated ticker in the text yields one span per occurrence.
    text = "AAPL up, AAPL down"
    response = {"entities": [{"text": "AAPL", "label": "ticker"}]}
    task, _ = auto_label.parse_response_to_task(text, response, 1)
    spans = [(r["value"]["start"], r["value"]["end"]) for r in task["annotations"][0]["result"]]
    assert spans == [(0, 4), (9, 13)]


def test_parse_response_keeps_longest_of_overlapping_variants():
    # "$ANPA" / "ANPA" overlap (cashtag + bare ticker); keep the wider "$ANPA".
    text = "I bought $ANPA today"
    response = {
        "entities": [
            {"text": "ANPA", "label": "ticker"},
            {"text": "$ANPA", "label": "ticker"},
        ]
    }
    task, _ = auto_label.parse_response_to_task(text, response, 1)
    results = task["annotations"][0]["result"]
    assert len(results) == 1
    assert results[0]["value"]["text"] == "$ANPA"


def test_parse_response_keeps_longest_company_variant():
    # Nested company variants collapse to the single longest span.
    text = "Rich Sparkle Limited, also called Rich Sparkle or just Rich, filed."
    response = {
        "entities": [
            {"text": "Rich", "label": "company"},
            {"text": "Rich Sparkle", "label": "company"},
            {"text": "Rich Sparkle Limited", "label": "company"},
        ]
    }
    task, _ = auto_label.parse_response_to_task(text, response, 1)
    results = task["annotations"][0]["result"]
    texts = sorted(r["value"]["text"] for r in results)
    # The leading "Rich Sparkle Limited" subsumes the nested variants at offset
    # 0; the standalone "Rich Sparkle" and "Rich" later in the text survive.
    assert "Rich Sparkle Limited" in texts
    starts = [r["value"]["start"] for r in results]
    assert 0 in starts  # the long variant is kept
    # No two kept spans overlap.
    intervals = sorted((r["value"]["start"], r["value"]["end"]) for r in results)
    assert all(intervals[i][1] <= intervals[i + 1][0] for i in range(len(intervals) - 1))



# --- _fuzzy_find ---------------------------------------------------------


def test_fuzzy_find_matches_normalised_company_name():
    # "Nvidia" (canonical) vs "Nvdias" (informal possessive without apostrophe)
    text = "Foxconns facility for Nvdias GB200 is in Guadalajara."
    results = auto_label._fuzzy_find("Nvidia", text)
    assert len(results) == 1
    start, end, matched = results[0]
    assert matched == "Nvdias"
    assert text[start:end] == "Nvdias"


def test_fuzzy_find_matches_possessive_company_name():
    # "Foxconn" vs "Foxconns" (s appended without apostrophe)
    text = "Foxconns facility for Nvdias GB200 is in Guadalajara."
    results = auto_label._fuzzy_find("Foxconn", text)
    assert len(results) == 1
    _, _, matched = results[0]
    assert matched == "Foxconns"


def test_fuzzy_find_returns_empty_for_unrelated_word():
    text = "The market rallied strongly today."
    results = auto_label._fuzzy_find("Nvidia", text)
    assert results == []


def test_fuzzy_find_skips_short_entities():
    # Entities shorter than 4 chars are excluded to avoid false positives.
    text = "AMD is a chipmaker."
    results = auto_label._fuzzy_find("AMX", text)
    assert results == []


# --- parse_response_to_task fuzzy fallback -------------------------------


def test_parse_response_fuzzy_nvdias_nvidia():
    # Reproduces the reported bug: LLM returns "Nvidia" but text has "Nvdias".
    text = (
        "I wish I knew where to ask...\n\n"
        "Foxconns facility for Nvdias GB200 is in Guadalajara."
    )
    response = {"entities": [{"text": "Nvidia", "label": "company"}]}
    task, dropped = auto_label.parse_response_to_task(text, response, 99)
    assert dropped == [], f"Entity was unexpectedly dropped: {dropped}"
    result = task["annotations"][0]["result"]
    assert len(result) == 1
    assert result[0]["value"]["text"] == "Nvdias"


def test_parse_response_fuzzy_foxconn_in_foxconns():
    # "Foxconn" is a literal substring of "Foxconns", so exact match succeeds
    # (offset 0-7) and no fuzzy lookup is needed.  Entity must not be dropped.
    text = "Foxconns facility for Nvdias GB200 is in Guadalajara."
    response = {"entities": [{"text": "Foxconn", "label": "company"}]}
    task, dropped = auto_label.parse_response_to_task(text, response, 1)
    assert dropped == []
    result = task["annotations"][0]["result"]
    assert len(result) == 1
    assert result[0]["value"]["text"] == "Foxconn"


def test_fuzzy_find_matches_word_with_extra_chars():
    # "Foxconn" is NOT an exact match for "Foxconns" when testing _fuzzy_find
    # directly (without the exact-match pre-check), so fuzzy should find it.
    text = "Foxconns facility in Guadalajara."
    results = auto_label._fuzzy_find("Foxconn", text)
    assert len(results) == 1
    _, _, matched = results[0]
    assert matched == "Foxconns"


def test_parse_response_fuzzy_does_not_match_unrelated():
    text = "The market rallied strongly today."
    response = {"entities": [{"text": "Nvidia", "label": "company"}]}
    task, dropped = auto_label.parse_response_to_task(text, response, 1)
    assert len(dropped) == 1
    assert dropped[0][2] == "not_found"


def test_run_interactive_retries_same_batch_on_bad_json(tmp_path):
    posts = [{"text": "AAPL"}]
    args = _args(tmp_path, batch_size=2)
    stdin = iter(
        [
            "not json at all",
            "END",
            # retry the SAME batch with valid JSON
            '{"entities": [{"text": "AAPL", "label": "ticker"}]}',
            "END",
        ]
    )
    auto_label.run_interactive(posts, args, line_source=stdin)

    data = json.loads(open(args.output, encoding="utf-8").read())
    assert [t["id"] for t in data] == [1000]
    assert data[0]["annotations"][0]["result"][0]["value"]["text"] == "AAPL"
