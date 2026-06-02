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
