"""Tests for the set-based, dedup-per-document error categorization."""


def _ctx(surface, context="...[x]..."):
    return {"surface": surface, "context": context}


def test_collect_gold_contexts_dedups_and_keeps_original_surface():
    from trainer.error_analysis import collect_gold_contexts

    dataset = [
        {
            "text": "GME GME gme $FUTU",
            "entities": [
                {"start": 0, "end": 3, "label": "ticker"},   # GME
                {"start": 4, "end": 7, "label": "ticker"},   # GME
                {"start": 8, "end": 11, "label": "ticker"},  # gme
                {"start": 12, "end": 17, "label": "ticker"}, # $FUTU
            ],
        }
    ]
    gold = collect_gold_contexts(dataset)
    # Three GME mentions collapse to one key; $FUTU normalizes to FUTU.
    assert set(gold[0]) == {("GME", "ticker"), ("FUTU", "ticker")}
    # Original surface is preserved (first seen) for downstream span-locating.
    assert gold[0][("GME", "ticker")]["surface"] == "GME"
    assert gold[0][("FUTU", "ticker")]["surface"] == "$FUTU"


def test_categorize_errors_splits_fp_fn_and_confusion():
    from trainer.error_analysis import categorize_errors

    gold_ctx = [{
        ("GME", "ticker"): _ctx("GME"),
        ("SOFI", "ticker"): _ctx("SOFI"),
        ("FUTU", "ticker"): _ctx("$FUTU"),
    }]
    pred_ctx = [{
        ("GME", "ticker"): _ctx("gme"),       # TP (normalized match)
        ("SOFI", "company"): _ctx("SOFI"),    # label confusion vs gold ticker
        ("HERE", "company"): _ctx("here"),    # pure FP
    }]
    dataset = [{"text": "doc text"}]

    cats = categorize_errors(pred_ctx, gold_ctx, dataset)

    assert [(r["text"], r["label"]) for r in cats["pure_fp"]] == [("here", "company")]
    assert [(r["text"], r["label"]) for r in cats["pure_fn"]] == [("$FUTU", "ticker")]
    assert len(cats["confusion"]) == 1
    conf = cats["confusion"][0]
    assert conf["text"] == "SOFI"
    assert conf["gold_label"] == "ticker"
    assert conf["pred_label"] == "company"
    # per-doc tallies use the raw (deduped) FP/FN set sizes
    assert cats["per_doc"][0]["n_fp"] == 2
    assert cats["per_doc"][0]["n_fn"] == 2
