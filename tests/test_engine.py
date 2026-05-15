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


def test_reddit_post(recognizer):
    text = """
    Positions in the second image. Please give yourself some grace, sell now eve during market hours if you are able, take your profits and just go outside and enjoy your day tomorrow knowing you have that in the bank. I was up 130k waiting for a stock to go up just a few more dollars and it never did. Quickly dropped. Went on tilt. Don't be like me. Take your profit.
    """
    results = recognizer.recognize_ai(text)
    assert len(results) == 0


def test_very_long_post(recognizer):
    text = r"""
Title:$BIRK May 13th Earnings DD: The Triple Tariff Catalyst. Why Illegal Taxes and Refund Claims make this a $55+ Stock.
Description: I put the TLDR at the top cause I know you regards won't read the bulk of this:

* Tariffs pressured this company's stock price throughout 2025, causing it to drop from its highs of $60+. Currently trading in $39-$40 range.
* Birkenstock lowered guidance in the December 2025 Earnings due to Tariffs and FX headwinds. Their all-in tariff rate should have been \~30% during the December 2025 and again in the February 2026 earnings call
* Since then, Supreme court ruled IEEPA illegal, they now qualify for IEEPA Tariff refunds, and recently, section 122 was ruled illegal by the trade courts
* Currently, the all-on tariff rate should be **\~20% at the maximum**, potentially only 8-10% if nothing replaces 122 that specifically targets European footwear.
* May 13th (BMO) will be the first time they will be able to address the now improved tariff situation since they lowered guidance in December.
* **This leads me to believe there are 3 catalysts going into May 13th earnings: tariff rate reduction, Tariff refund claim disclosure, guidance revision**.
* This is a luxury apparel growth stock trading like a value stock because it has been highly shorted due to tariffs eating into margins. And now those tariffs might go away entirely, plus a refund for past tariffs.
* If management signals tariff headwinds won't be as bad as previously guided, the narrative shifts to it being a growth stock. Under that scenario, a 19 forward P/E is much more reasonable and that puts BIRK at $55.

Full DD:

**Guidance Revision**

At their December earnings, management guided FY2026 with a 100bps tariff headwind baked into both gross margin and EBITDA margin guidance that assumed the \~30% all-in rate persisting through the year. **That assumption is now wrong by at least 10 percentage points.** They haven't had a chance to update it publicly.

February's call maintained full-year guidance because the SCOTUS ruling happened eight days later and Section 122 took effect four days after that. Management had no basis to update numbers on the Feb call. May 13th is the first opportunity to do so, and I don't think it's been priced in due to how thinly traded and overly shorted this stock got during the Iran war.

If the 100bps tariff headwind assumption shrinks to reflect the lower rate, that flows directly into updated gross margin and EBITDA margin guidance. America is their largest market, so the margin improvement will be substantial. They also have a strong record of beating the streets estimates, so I hope a strong beat tomorrow means they lift their guidance. 

**IEEPA refund disclosure**

BIRK almost certainly qualifies for IEEPA tariff refunds. CBP opened the CAPE refund portal on April 20th. The refundable amount is the IEEPA-attributable layer (roughly 10 percentage points) on all German imports entered between April 2025 and February 24, 2026. Other footwear companies disclosed CAPE filings on the day the portal opened. Analysts will ask BIRK the same question tomorrow.

**Section 122 now Illegal too**

The CIT ruled Section 122 unlawful on May 7. The injunction only covers named plaintiffs (Burlap & Barrel, Basic Fun), so BIRK is still paying the 10% rate. But the legal foundation for a future refund claim now exists. Management will be asked whether they're preserving entry documentation to protect those rights. Given the July 24 expiry of Section 122 regardless, and the political impossibility of congressional extension, the 10% rate is likely gone after July 24. Their rate would then drop to 8-10%, from its current \~20% rate, which is what their tariff rate was pre-liberation day. The risk here is that future tariffs may increase it back to 20%, but I have reason to believe that the next round of tariff attacks may not even touch Birkenstock.

**Why I think they will overcome future tariff pressures:**

Trump has specifically directed his recent tariff threats at EU steel, aluminum and autos, not apparel. EU footwear and leather goods have faced no equivalent product specific action beyond the now illegal IEEPA reciprocal rate. The Section 301 investigation initiated in March 2026 does include the EU, but it targets structural excess manufacturing capacity, aka countries/industries that have cheap manufacturing where USA has bigger deficit. That means apparel companies with Chinese and Southeast Asian factories will get hit. EU autos and steels will get hit, but German made sandals? Not on really on this administrations radar based on any of its messaging. Their direct footwear peers (ONON, Deckers, Nike) all are exposed to Asian manufacturing tariffs, while Birkenstock manufactures 95% of their product in Germany. That gives them a meaningful advantage to shake off its current tariff-based valuations.

Birkenstock is approaching May 13 with a lower all-in rate, a refund claim for overpaid duties, and a manufacturing geography that does not appear to be in the crosshairs of Trump’s next wave of tariffs. If the tariff story really changes the way I see it changing, its share price should no longer be suppressed, its valuations should reflect that it is a growth stock, not a value stock.

**Growth Prospects:**

On the demand side, Birkenstock's growth trajectory has been self-evidently strong, FY2025 delivered record revenue of €2.1 billion, up 18% in constant currency, with Americas growing 18% in constant currency. Demand signal holds up in search data too: Google Trends data shows "Birkenstock clogs" maintaining consistent baseline interest throughout the year with a significant spike into the December 2025 holiday season, which analysts have noticed. They are rapidly expanding in Asia, their production is ramping up and they're growing their direct-to-consumer sales (which is a high margin business). These are very positive tailwinds for growth, completely independent of tariffs.

The macro backdrop also skews in Birkenstock's favor relative to most of its apparel and footwear peers. The US consumer economy is visibly K-shaped, with premium brands like Ralph Lauren, Tapestry, American Express, United, and Delta all exceeding expectations as their affluent customer bases spent freely on high-margin goods, while value-focused brands serving lower-income households reported pullbacks. Birkenstock sits firmly in the first camp. We all know we are in a K-shaped economy, and Birkenstock is a benefiter of that. Recently, it traded down with other retail apparel and footwear stocks during the Iran war, when it really shouldn’t. They have solid demand that is not going away, it should be priced similar to Ralph Lauren.

**Valuation:**

Per Fintel, Birkenstock had a reported \~19% short float and \~6.8 days to cover. This strikes me as high for a company that has 18% revenue growth and real growing profitability. The primary reason for it being shorted has to be tariffs. BIRK’s forward P/E, and PEG are clearly very suppressed compared to industry standard for Apparel Stock P/E’s especially compared to companies like Nike that are disadvantaged by their China/southeast Asia manufacturing.

Birk’s current forward PE is 13.67 and its PEG is at 0.87 (per finviz). If the narrative around Birkenstocks tariff woes changes after tomorrows earning call, these ratios are insane discounts. A re-pricing of Birkenstocks share price would send it soaring. As of January 2026, apparel stocks had a forward P/E of 24.89 and a PEG of 1.99 (Per NYU study: [Price Earnings Ratios](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/pedata.html)). **Birkenstock simply being priced at those ratios would put the stock price at $71- $89.** I don’t think it will go that high immediately post earnings, but **the average analyst values this stock at $57.** Further, **Michael Burry is buying this stock too**. It’s clearly undervalued, and May 13th earnings calls has enough potential catalysts to reverse its year-long bearish downtrend (that reversal is already starting to form in its graph).

A narrative shift from the tariff pressures could send this stock soaring. **My personal price target is $55, which would put it at a Forward P/E of 19 and a PEG of 1.22.** By comparison, Nike, a company with significantly more tariff and growth pressure, trades at forward P/E of \~23 and a PEG of \~8.46. Ralph Lauren, a fellow luxury apparel stock, trades at 19 forward P/E and 1.08 PEG. By current street predictions, Birkenstock is expected to grow earnings by 18%, whereas Ralph is expected to grow 10%. But BIRK has better debt ratios than Ralph Lauren (3.13 vs 2.1 and 0.49 vs .99 for current ratio and debt/equity ratio respectively). For you regards who don't understand that, it has much less debt than Ralph Lauren, while growing faster with cheaper valuations. This part can go on a lot longer and it's obviously very subjective, but a forward PE of 19 and a PEG of 1.22 for a company that's growing and already profitable with low debt really doesn’t sound so farfetched, eh?

Disclaimer: I don't expect a 40% move in one day, this not a crowded stock where that can happen. If it transitions from non-crowded to crowded, it's possible, but I'm not counting on it.

Currently, I think a move up to $45 post earnings is more likely if they post good earnings beat and my catalysts materialize on the earnings call. I would expect it to run higher once trump does his weekly taco. Stock rotation is buzzing in the headlines today too, which would help allocate some flows to mid-cap consumer cyclical industries stocks that are showing some promise. 

**The Bottom Line**

Unlike other meme stocks on here, Birkenstock has real growth and proven profitability. $39 is simply too cheap by a lot of metrics. This is a rare find and strong buying opportunity. Given that tmrw will be the first time Birkenstocks management will address the improved tariff situation, these catalysts have a real chance to materialize. If that happens, I expect the narrative to shift from tariffs to growth, and there’s no way this stock stays at the $39 range for much longer.

Do your own research. I have a horrible track record, but I think I'm really on to something this time.

Positions:

https://preview.redd.it/ur547m9ndp0h1.png?width=1195&format=png&auto=webp&s=3d6a56448eaa6f626b3c7cbf185a14811610ed41

* Note: I did make $6.5K on $42 Jul calls and then rolled to the $45 calls. In retrospect, should have taken profit and re-entered on the pullback, but live and learn.
* Plan is sell 20-30 contracts after earnings are released and let the rest run for another week or two depending on the guidance and overall market movements.
* If I were to enter this trade right now, instead of weeks ago, I would consider the $37.5 or $40 May calls and then sell them and buy shares to hold for $55+ (NOT FINANCIAL ADVICE)
"""
    results = recognizer.recognize_ai(text)
    assert "BIRK" in results
    assert "DECK" in results
    assert "NKE" in results
    assert "ONON" in results
    assert "RL" in results
    assert "TPR" in results
    assert "DAL" in results
    assert len(results) == 7


def test_common_slang(recognizer):
    text = "My DD is that this is an ITM play, YOLO!"
    results = recognizer.recognize_ai(text)
    assert "DD" not in results
    assert "ITM" not in results
    assert "YOLO" not in results
    assert len(results) == 0


def test_ethe_post(recognizer):
    text = "Grayscale Trust Selling at a huge discount. will climb when it forks to proof of stake. ETHE will get both forks, will probably sell off the proof of work stake and pay a special dividend. Then the trust will most likely start paying a dividend when it gets paid for its proof of stake. Could be 12% or more per year. Plus ETHE is heavily shorted. This is rocketing higher this month. No doubts."
    results = recognizer.recognize_ai(text)
    assert "ETHE" in results
    assert len(results) == 1
