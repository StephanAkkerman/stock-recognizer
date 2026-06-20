"""Tests for trainer/benchmark.py engine evaluation path."""

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build a minimal fake StockRecognizer
# ---------------------------------------------------------------------------

def _make_fake_engine(recognize_ai_results):
    """Return a mock StockRecognizer whose recognize_ai() returns the given
    per-call results (one list per document, in order)."""
    engine = MagicMock()
    engine.valid_tickers = {"AAPL", "TSLA", "MSFT"}
    engine.company_to_ticker = {"APPLE": "AAPL", "TESLA": "TSLA"}

    def _clean_token(token):
        return token.upper().strip().replace("$", "")

    engine._clean_token = _clean_token
    engine.recognize_ai.side_effect = recognize_ai_results
    return engine


# ---------------------------------------------------------------------------
# Unit tests for _resolve_gold_tickers
# ---------------------------------------------------------------------------

def test_resolve_gold_tickers_direct_ticker():
    """A gold span that is itself a valid ticker resolves directly."""
    from trainer.benchmark import _resolve_gold_tickers

    engine = _make_fake_engine([])
    entry = {
        "text": "Buy AAPL now",
        "entities": [{"start": 4, "end": 8, "label": "ticker"}],
    }
    result = _resolve_gold_tickers(engine, entry)
    assert result == frozenset({"AAPL"})


def test_resolve_gold_tickers_company_resolution():
    """A gold span that is a company name resolves through company_to_ticker."""
    from trainer.benchmark import _resolve_gold_tickers

    engine = _make_fake_engine([])
    entry = {
        "text": "Apple is doing well",
        "entities": [{"start": 0, "end": 5, "label": "company"}],
    }
    result = _resolve_gold_tickers(engine, entry)
    assert result == frozenset({"AAPL"})


def test_resolve_gold_tickers_unresolvable_span():
    """A span that cannot be resolved to any ticker is silently dropped."""
    from trainer.benchmark import _resolve_gold_tickers

    engine = _make_fake_engine([])
    engine.valid_tickers = set()
    engine.company_to_ticker = {}
    entry = {
        "text": "Some unknown thing",
        "entities": [{"start": 0, "end": 4, "label": "company"}],
    }
    result = _resolve_gold_tickers(engine, entry)
    assert result == frozenset()


# ---------------------------------------------------------------------------
# Unit tests for engine_evaluate_model (2-document dataset)
# ---------------------------------------------------------------------------

TINY_DATASET = [
    # Doc 0: gold = {AAPL}, pred = {AAPL} → TP=1 FP=0 FN=0
    {
        "text": "Buy AAPL now",
        "entities": [{"start": 4, "end": 8, "label": "ticker"}],
    },
    # Doc 1: gold = {TSLA}, pred = {TSLA, MSFT} → TP=1 FP=1 FN=0
    {
        "text": "Tesla is rising, TSLA to the moon",
        "entities": [{"start": 0, "end": 5, "label": "company"}],
    },
]


def test_engine_evaluate_model_tp_fp_fn():
    """engine_evaluate_model computes TP/FP/FN correctly across 2 documents."""
    from trainer.benchmark import engine_evaluate_model
    import stock_recognizer.engine as _eng_mod

    fake_engine = _make_fake_engine(
        recognize_ai_results=[["AAPL"], ["TSLA", "MSFT"]]
    )

    original = _eng_mod.StockRecognizer
    _eng_mod.StockRecognizer = lambda **kw: fake_engine
    try:
        result = engine_evaluate_model("fake/path", TINY_DATASET)
    finally:
        _eng_mod.StockRecognizer = original

    overall = result["overall"]
    # Doc 0: gold={AAPL}, pred={AAPL}         → TP=1, FP=0, FN=0
    # Doc 1: gold={TSLA}, pred={TSLA, MSFT}   → TP=1, FP=1, FN=0
    # Total: TP=2, FP=1, FN=0  → P=2/3, R=1.0, F1=0.8
    assert overall["p"] == pytest.approx(2 / 3, abs=1e-6)
    assert overall["r"] == pytest.approx(1.0, abs=1e-6)
    assert overall["f1"] == pytest.approx(0.8, abs=1e-6)


def test_engine_evaluate_model_returns_overall_key():
    """engine_evaluate_model always returns a dict with an 'overall' key."""
    from trainer.benchmark import engine_evaluate_model
    import stock_recognizer.engine as _eng_mod

    fake_engine = _make_fake_engine(recognize_ai_results=[[], []])

    original = _eng_mod.StockRecognizer
    _eng_mod.StockRecognizer = lambda **kw: fake_engine
    try:
        result = engine_evaluate_model("fake/path", TINY_DATASET)
    finally:
        _eng_mod.StockRecognizer = original

    assert "overall" in result
    assert set(result["overall"].keys()) == {"p", "r", "f1"}


# ---------------------------------------------------------------------------
# Set-based, dedup-per-document scoring (matches the engine's "caught once"
# contract: recognize() returns a *set* of tickers, not per-occurrence spans).
# ---------------------------------------------------------------------------

def test_normalize_entity_folds_cashtag_case_and_whitespace():
    from trainer.benchmark import normalize_entity

    assert normalize_entity("$GME") == "GME"
    assert normalize_entity("gme") == "GME"
    assert normalize_entity("  GME ") == "GME"
    assert normalize_entity("$ AUG") == "AUG"
    assert normalize_entity("AMC Theatres") == "AMC THEATRES"


def test_prepare_eval_inputs_dedups_repeated_mentions():
    """A ticker mentioned many times in one doc collapses to a single gold key."""
    from trainer.benchmark import prepare_eval_inputs

    dataset = [
        {
            "text": "GME GME gme to the moon",
            "entities": [
                {"start": 0, "end": 3, "label": "ticker"},
                {"start": 4, "end": 7, "label": "ticker"},
                {"start": 8, "end": 11, "label": "ticker"},
            ],
        }
    ]
    _, _, gold_per_doc, gold_by_label = prepare_eval_inputs(dataset, ["ticker", "company"])

    assert gold_per_doc[0] == {("GME", "ticker")}
    assert gold_by_label[0]["ticker"] == {("GME", "ticker")}
    assert gold_by_label[0]["company"] == set()


class _FakeGmeModel:
    """Fake GLiNER2 that tags every case-insensitive 'gme' occurrence per chunk."""

    def batch_extract_entities(self, texts, labels, **kwargs):
        outputs = []
        for t in texts:
            items = []
            low = t.lower()
            i = low.find("gme")
            while i != -1:
                items.append({"start": i, "end": i + 3})
                i = low.find("gme", i + 1)
            outputs.append({"entities": {"ticker": items}})
        return outputs


def test_evaluate_model_dedups_repeated_mentions_to_one_tp():
    """Catching GME many times in a doc counts as a single TP, no FP/FN."""
    from trainer.benchmark import evaluate_model, prepare_eval_inputs

    dataset = [
        {
            "text": "GME GME gme are all the same ticker",
            "entities": [
                {"start": 0, "end": 3, "label": "ticker"},
                {"start": 4, "end": 7, "label": "ticker"},
                {"start": 8, "end": 11, "label": "ticker"},
            ],
        }
    ]
    flat_chunks, ranges, gold_per_doc, gold_by_label = prepare_eval_inputs(
        dataset, ["ticker", "company"]
    )

    scores = evaluate_model(
        _FakeGmeModel(),
        flat_chunks,
        ranges,
        gold_per_doc,
        gold_by_label,
        label_descriptions={"ticker": "x", "company": "y"},
    )
    assert scores["overall"]["r"] == pytest.approx(1.0)
    assert scores["overall"]["p"] == pytest.approx(1.0)
    assert scores["ticker"]["f1"] == pytest.approx(1.0)
