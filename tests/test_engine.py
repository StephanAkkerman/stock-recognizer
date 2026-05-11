import pytest

from stock_recognizer import StockRecognizer


@pytest.fixture(scope="module")
def recognizer():
    # We initialize once for the whole test session to save time
    return StockRecognizer(use_ai=True)


def test_basic_cashtags(recognizer):
    text = "I am long on $AAPL and $MSFT"
    results = recognizer.recognize(text)
    assert "AAPL" in results
    assert "MSFT" in results
    assert len(results) == 2


def test_slang_blacklist(recognizer):
    text = "My DD is that this is an ITM play, YOLO!"
    results = recognizer.recognize(text)
    # None of these should be caught as tickers despite being uppercase
    assert "DD" not in results
    assert "ITM" not in results
    assert "YOLO" not in results


def test_yelling_protection(recognizer):
    text = "PLEASE SERVICE NOW PLEASE I NEED THIS TO HIT 110"
    results = recognizer.recognize(text)
    # 'SERVICE', 'NOW', 'PLEASE' are all tickers, but should be ignored
    # because the post is mostly uppercase and lacks cashtags.
    assert "SERVICE" not in results
    assert "PLEASE" not in results
    # '110' should be ignored because it's not alphabetic
    assert "110" not in results


def test_company_mapping(recognizer):
    # This requires AI and the mapping dictionary
    text = "TSMC is the best semiconductor supplier for Apple"
    results = recognizer.recognize_ai(text)
    assert "TSM" in results
    assert "AAPL" in results


def test_possessives_and_plurals(recognizer):
    text = "I like AAPLs chart and MSFT's dividends"
    # This test might fail initially! It tells us we need to handle trailing 'S
    results = recognizer.recognize(text)
    assert "AAPL" in results
    assert "MSFT" in results


def test_complex_wsb_post(recognizer):
    text = "$PLAB DD: easy to understand TSMC supplier chip tools trade"
    results = recognizer.recognize_ai(text)
    assert "PLAB" in results
    assert "TSM" in results
    assert "ASP" not in results  # Check for the hallucination we saw earlier


def test_invalid_filings(recognizer):
    text = "$MVST DD: US battery company carving out assets (see 10-K)"
    results = recognizer.recognize(text)
    assert "MVST" in results
    assert "10-K" not in results


def test_lowercase_and_mixed_case(recognizer):
    # Redditors often type fast on mobile, ignoring caps lock
    text = "I love $aapl and think $TsLa will moon."
    results = recognizer.recognize(text)
    assert "AAPL" in results
    assert "TSLA" in results


def test_punctuation_and_formatting(recognizer):
    # Tickers are often wrapped in brackets, quotes, or followed by commas
    text = "Big tech (MSFT, GOOGL) is carrying the market... maybe NVDA too!?"
    results = recognizer.recognize(text)
    assert "MSFT" in results
    assert "GOOGL" in results
    assert "NVDA" in results


def test_ai_company_mapping_edge_cases(recognizer):
    # Testing if the AI can handle natural language without any tickers present
    text = "I think Amazon is going to crush earnings. Also betting heavily on Meta."
    results = recognizer.recognize_ai(text)
    assert "AMZN" in results
    assert "META" in results


def test_url_handling(recognizer):
    # We want to make sure it doesn't crash on URLs, but CAN extract from them if a ticker is present
    text = "Read the news here: https://finance.yahoo.com/quote/AMD/news/"
    results = recognizer.recognize(text)
    assert "AMD" in results


def test_long_reddit_dd_simulation(recognizer):
    # The Ultimate Stress Test: A realistic, messy WallStreetBets post
    text = """
    Alright degenerates, gather round for some real DD. 
    I've been reading the 10-K and 10-Q filings for Palantir ($pltr) and the YoY growth is insane. 
    The CEO knows exactly what he is doing. Last year I YOLO'd my life savings into GME and AMC, 
    but now I'm playing it safe.
    
    I'm buying shares of Wendy's (WEN) because I literally work behind the dumpster there. 
    Are we in a bubble? IS THE FED GOING TO PIVOT? Who knows. 
    
    TLDR: PLTR to the moon. Don't buy OTM options on random penny stocks. HODL your AAPL.
    """

    results = recognizer.recognize_ai(text)

    # --- The "Must Have" Tickers ---
    assert "PLTR" in results
    assert "GME" in results
    assert "AMC" in results
    assert "WEN" in results
    assert "AAPL" in results

    # --- The "Must NOT Have" Noise (Checking our Blacklist & Logic) ---
    assert "DD" not in results
    assert "YOY" not in results
    assert "CEO" not in results
    assert "YOLO" not in results
    assert "OTM" not in results
    assert "TLDR" not in results
    assert "HODL" not in results
    assert "FED" not in results
    assert "10-K" not in results
    assert "10-Q" not in results
    assert "IS" not in results  # from the all-caps sentence
    assert "THE" not in results
