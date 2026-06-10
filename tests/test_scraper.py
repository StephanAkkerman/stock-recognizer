import asyncio
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import scraper  # noqa: E402


def _sub(sid, *, is_self=True, selftext="", title="", **extra):
    return types.SimpleNamespace(
        id=sid, is_self=is_self, selftext=selftext, title=title, **extra
    )


# --- submission_text -----------------------------------------------------


def test_submission_text_self_ok():
    text, reason = scraper.submission_text(
        _sub("a", selftext="x" * 80), min_chars=50, max_chars=2000
    )
    assert reason is None
    assert text == "x" * 80


def test_submission_text_self_too_short():
    text, reason = scraper.submission_text(
        _sub("a", selftext="short"), min_chars=50, max_chars=2000
    )
    assert text is None
    assert reason == "too_short"


def test_submission_text_self_too_long():
    text, reason = scraper.submission_text(
        _sub("a", selftext="x" * 50), min_chars=10, max_chars=20
    )
    assert text is None
    assert reason == "too_long"


def test_submission_text_no_max_cap():
    text, reason = scraper.submission_text(
        _sub("a", selftext="x" * 9999), min_chars=10, max_chars=None
    )
    assert reason is None
    assert len(text) == 9999


def test_submission_text_link_skipped_by_default():
    # Default is include_link_titles=False: bodyless link/image posts (whose only
    # text would be the title) are dropped rather than duplicated into `text`.
    text, reason = scraper.submission_text(
        _sub("a", is_self=False, title="YOLO 100k $GME calls to the moon baby"),
        min_chars=50,
        max_chars=2000,
    )
    assert text is None
    assert reason == "link_post"


def test_submission_text_link_title_captured_when_enabled():
    title = "YOLO 100k $GME calls"
    text, reason = scraper.submission_text(
        _sub("a", is_self=False, title=title),
        min_chars=50,  # selftext floor does not apply to titles
        max_chars=2000,
        include_link_titles=True,
        min_title_chars=10,
    )
    assert reason is None
    assert text == title


def test_submission_text_link_title_too_short():
    text, reason = scraper.submission_text(
        _sub("a", is_self=False, title="$GME"),
        min_chars=50,
        max_chars=2000,
        include_link_titles=True,
        min_title_chars=10,
    )
    assert text is None
    assert reason == "too_short"


# --- build_search_queries ------------------------------------------------


def test_build_search_queries_includes_defaults_and_tickers():
    queries = scraper.build_search_queries(tickers=["GME", "TSLA"])
    assert "GME" in queries
    assert "TSLA" in queries
    # default keyword set is present
    assert any(q.lower() == "calls" for q in queries)


def test_build_search_queries_dedups_case_insensitively():
    queries = scraper.build_search_queries(
        keywords=["DD", "dd", "calls"], tickers=["GME", "gme"]
    )
    lowered = [q.lower() for q in queries]
    assert lowered.count("dd") == 1
    assert lowered.count("gme") == 1


def test_build_search_queries_preserves_order():
    queries = scraper.build_search_queries(keywords=["alpha", "beta"], tickers=["GME"])
    assert queries == ["alpha", "beta", "GME"]


# --- _ScrapeState.add ----------------------------------------------------


def _state(**kw):
    defaults = dict(
        known_ids=set(),
        known_text_hashes=set(),
        output_file=None,
        save_every=50,
        min_chars=50,
        max_chars=2000,
        include_link_titles=True,
        min_title_chars=10,
    )
    defaults.update(kw)
    return scraper._ScrapeState(**defaults)


def test_state_add_appends_valid_post():
    st = _state()
    assert st.add(_sub("a", selftext="x" * 80)) is True
    assert len(st.posts) == 1
    assert st.posts[0]["id"] == "a"


def test_state_add_dedups_by_id():
    st = _state(known_ids={"a"})
    assert st.add(_sub("a", selftext="x" * 80)) is False
    assert st.counters["dup_id"] == 1
    assert st.posts == []


def test_state_add_dedups_by_text_hash():
    st = _state()
    st.add(_sub("a", selftext="x" * 80))
    # different id, identical text -> dropped as text dup
    assert st.add(_sub("b", selftext="x" * 80)) is False
    assert st.counters["dup_text"] == 1
    assert len(st.posts) == 1


def test_state_add_counts_link_post_when_disabled():
    st = _state(include_link_titles=False)
    assert st.add(_sub("a", is_self=False, title="t" * 80)) is False
    assert st.counters["link_post"] == 1


def test_state_add_captures_link_title_when_enabled():
    st = _state()
    assert st.add(_sub("a", is_self=False, title="YOLO 100k $GME calls")) is True
    assert st.posts[0]["text"] == "YOLO 100k $GME calls"


def test_state_flush_writes_csv(tmp_path):
    out = str(tmp_path / "posts.csv")
    st = _state(output_file=out, save_every=1000)
    st.add(_sub("a", selftext="x" * 80, title="T", score=1, url="u", created_utc=1.0))
    st.flush()
    df = pd.read_csv(out)
    assert list(df["id"]) == ["a"]


# --- _collect (async drain) ----------------------------------------------


class _FakeAiter:
    """Minimal async-iterable over a fixed list of submissions."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        async def gen():
            for it in self._items:
                yield it

        return gen()


def test_collect_stops_at_target_and_dedups_across_sources():
    async def run():
        scraper_obj = scraper.RedditScraper()
        st = _state()
        # Two sources; "a" repeats across both and must only count once.
        sources = [
            ("src1", _FakeAiter([_sub("a", selftext="a" * 80), _sub("b", selftext="b" * 80)])),
            ("src2", _FakeAiter([_sub("a", selftext="a" * 80), _sub("c", selftext="c" * 80)])),
        ]
        posts = await scraper_obj._collect(sources, st, target=2, desc="test")
        return posts, st

    posts, st = asyncio.run(run())
    assert len(posts) == 2  # stopped at target before reaching "c"
    assert [p["id"] for p in posts] == ["a", "b"]
