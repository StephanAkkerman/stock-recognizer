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


def test_reddit_noise_cleanup(recognizer):
    # Tests 'S', 'AM', 'HI', 'ADD' from your output
    text = "It’s almost Monday! I am getting ready. The CEO ate his hat. Robinhood adds short selling."
    results = recognizer.recognize(text)
    # These words are valid tickers in the DB but should be blocked by our Ambiguity Shield
    assert "S" not in results
    assert "AM" not in results
    assert "HI" not in results
    assert "ADD" not in results


def test_international_suffix_removal(recognizer):
    # Tests 'N1DA34.SA' and '7195.KL'
    text = "Nasdaq winners N1DA34.SA and Intel SK Hynix 7195.KL"
    results = recognizer.recognize(text)
    assert "N1DA34.SA" not in results
    assert "7195.KL" not in results


def test_url_and_artifact_filter(recognizer):
    # Tests 'WWW', 'EYES', 'AMP'
    text = "Source: https://www.techpowerup.com/sk-hynix-eyes-intel-amp"
    results = recognizer.recognize(text)
    assert "WWW" not in results
    assert "EYES" not in results
    assert "AMP" not in results


def test_all_caps_sentence_protection(recognizer):
    # Tests 'IS', 'THE', 'FED'
    text = "IS THE FED GOING TO PIVOT? Who knows."
    results = recognizer.recognize(text)
    assert "IS" not in results
    assert "THE" not in results
    assert "FED" not in results


def test_new_company_seeds(recognizer):
    # Tests the updated seeds for Micron and Canon
    case1 = "Deutsche Bank raises Micron stock price target to $1,000"
    results1 = recognizer.recognize_ai(case1)
    assert "MU" in results1

    case2 = "The Lithography Canon $CAJPY"
    results2 = recognizer.recognize_ai(case2)
    assert "CAJPY" in results2


def test_reddit_noise_filter(recognizer):
    # Tests the specific noise words found in your last run
    text = "It’s almost Monday! I am getting ready. The CEO ate his hat. Robinhood adds short selling."
    results = recognizer.recognize(text)

    # These are all valid tickers that should be blocked by the Ambiguity Shield
    noise = ["AM", "HI", "S", "ADD"]
    for word in noise:
        assert word not in results


def test_financial_jargon_shield(recognizer):
    # Tests common financial terms that are also tickers
    text = "I am currently watching the AI and MIN futures. UPS is up. PSA: keep a cool head."
    results = recognizer.recognize(text)

    # 'AI' (C3.ai) and 'UPS' (United Parcel Service) are real tickers,
    # but in this context they are noise/words.
    # Without a $, they should be ignored.
    assert "AI" not in results
    assert "MIN" not in results
    assert "UPS" not in results
    assert "PSA" not in results


def test_international_denial(recognizer):
    # Tests that .SA and .KL and other suffixes are totally ignored
    text = "Nasdaq winners N1DA34.SA and Intel 7195.KL"
    results = recognizer.recognize(text)
    assert "N1DA34.SA" not in results
    assert "7195.KL" not in results


def test_lowercase_cashtag(recognizer):
    # Tests that $aapl works just as well as $AAPL
    text = "I am buying more $mram and $rklb"
    results = recognizer.recognize(text)
    assert "MRAM" in results
    assert "RKLB" in results


def test_complex_description_parsing(recognizer):
    # A simulated messy post body
    text = """
    Iran war could prompt Federal Reserve to raise rates, Pimco says.
    This is not financial advice, just my plan. I made it back to my peak.
    Some continued oil plays and SPY 0DTEs. Shift from AI to SPACE.
    Focus on Arxis, Inc. (ARXS).
    """
    results = recognizer.recognize_ai(text)

    # Arxis is a real ticker (ARXS) and has 'Inc' in the name, testing mapping
    assert "ARXS" in results

    # Ensure noise words used in this specific text are blocked
    blocked = ["BY", "SAY", "PLAN", "BACK", "PEAK", "SOME", "AI", "SO", "BE"]
    for word in blocked:
        assert word not in results


def test_cashtag(recognizer):
    text = "£500,000 on $MBLY 🤖 Good enough for Intel good enough for me. 5x my position after discussing this name with you degens over the weekend. Let’s ride. 😆"
    results = recognizer.recognize_ai(text)
    assert "MBLY" in results
    assert "INTC" in results
    assert len(results) == 2


def test_simple_tickers(recognizer):
    text = "RDDT and NOK gains Held over the weekend. Just getting started 🔥"
    results = recognizer.recognize_ai(text)
    assert "RDDT" in results
    assert "NOK" in results
    assert len(results) == 2


def test_urls(recognizer):
    text = """Title:Nasdaq’s top 10 winners averaged +784% gains, surpassing the +622% dot-com peak leaders before the 2000 crash
Description: Source: [https://finance.yahoo.com/markets/article/the-nasdaqs-top-winners-are-now-running-hotter-than-in-2000-chart-of-the-day-122715863.html](https://finance.yahoo.com/markets/article/the-nasdaqs-top-winners-are-now-running-hotter-than-in-2000-chart-of-the-day-122715863.html)"""
    results = recognizer.recognize_ai(text)
    assert len(results) == 0


def test_long_form_post(recognizer):
    text = """
    Title:Eugene's back in the game with a high conviction bet: KEEL
Description: Hi Regards, been a while. Well, I'm getting back in the game a bit.

This whole AI frothy space can be pretty tricky to navigate, and hardware is already way overpriced, and who the fuck knows what's happening with software in this environment... but one thing is clear: There is insatiable hunger for data center infrastructure.

Many players that are further ahead in this space have already had massive run-ups: NBIS, HUT, VRT.

**Why KEEL**? Because they are still early in their transformation from running crypto data centers to their AI datacenter pivot. Market cap is still relatively low at $2.5BB.

They just had their latest ER in which they articulated that they already have 2 Gigawatts of energy secured across north-american data center sites. These sites can go live as early as 2027... and they are looking to close 3 lease deals by end of 2026. As soon as they announce any one of these deals, I expect the stock to moon. They have enough cash and liquidity on hand through 2028 - well enough time to secure said lease deals. This means dilution events are unlikely.

So, just look at the charts for NBIS and HUT as their closest analogs. If they get all those deals signed, I expect this stock to 2-3x by end of year, and a 10x by end of 2028 once they actually start operating.

This is free money.

**Positions:**

10k shares

100 Jan 15 2027 $5 calls
    """
    results = recognizer.recognize_ai(text)
    assert "KEEL" in results
    assert "NBIS" in results
    assert "HUT" in results
    assert "VRT" in results
    assert len(results) == 4
