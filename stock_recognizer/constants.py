EXCHANGE_BLACKLIST = {
    ".SA",
    ".KL",
    ".F",
    ".TI",
    ".TO",
    ".L",
    ".DE",
    ".KL",
    ".KS",
    ".HK",
}

# Major exchange codes used by financedatabase for ticker-universe filtering.
# US major venues are enabled by default. Global majors can be opt-in.
US_MAJOR_EXCHANGES = {
    "NYQ",  # NYSE
    "NMS",  # NASDAQ Global Select
    "NGM",  # NASDAQ Global Market
    "NCM",  # NASDAQ Capital Market
    "ASE",  # NYSE American
}

GLOBAL_MAJOR_EXCHANGES = {
    "LSE",  # London Stock Exchange
    "JPX",  # Japan Exchange Group (Tokyo)
    "HKG",  # Hong Kong Exchange
    "ASX",  # Australian Securities Exchange
    "TOR",  # Toronto Stock Exchange
}

COMPANY_SEEDS = {
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "TSMC": "TSM",
    "TAIWAN": "TSM",
    "NVIDIA": "NVDA",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "FACEBOOK": "META",
    "NETFLIX": "NFLX",
    "TESLA": "TSLA",
    "PALANTIR": "PLTR",
    "GAMESTOP": "GME",
    "AMC": "AMC",
    "WENDY'S": "WEN",
    "WENDYS": "WEN",
    "WENDY": "WEN",
    "ROBINHOOD": "HOOD",
    "SNOWFLAKE": "SNOW",
    "MICRON": "MU",
    "CANON": "CAJPY",
    "INTEL": "INTC",
    "BLACKROCK": "BLK",
}

AMBIGUOUS_WORDS = {
    # Single Letters (Still keep these for AI validation)
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    # Months
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
    # Newly discovered Reddit noise from your output
    "BE",
    "AIN",
    "FUND",
    "DAN",
    "AM",
    "PSA",
    "AD",
    "AS" "MIN",
    "UPS",
    "HI",
    "AI",
    "EP",
    "AIM",
    "GAIN",
    "LUCK",
    "PURE",
    "LAD",
    "SAFE",
    "BACK",
    "PLAY",
    "TACO",
    "VERY",
    "BULL",
    "PLAN",
    "WELL",
    "RIDE",
    "GOOD",
    "YOU",
    "SEE",
    "SOME",
    "ANY",
    "OWN",
    "POST",
    "VIEW",
    "MOVE",
    "LIFE",
    "LINE",
    "WORK",
    "BOTH",
    "WWW",
    "HTTP",
    "HTTPS",
    "COM",
    # Slang / Finance
    "DD",
    "DYOR",
    "NFA",
    "GMV",
    "DTC",
    "FINRA",
    "HUGE",
    "FOUR",
    "RE",
    "PLUS",
    "AS",
    "WAVE",
    "PRE",
    "PEG",
    "EU",
    "BMO",
    "FORM",
    "RARE",
    "BASE",
    "CIT",
    "BEAT",
    "EH",
    "LOT",
    "VS",
    "AGO",
    "AKA",
    "PRKS",
    "WSB",
    "YOLO",
    "FOMO",
    "TLDR",
    "LFG",
    "MOASS",
    "HODL",
    "GUH",
    "ITM",
    "ATM",
    "OTM",
    "IV",
    "FD",
    "LEAPS",
    "PTE",
    "YOY",
    "QOQ",
    "EOD",
    "AH",
    "PM",
    "ATH",
    "CPI",
    "GDP",
    "EST",
    "CEO",
    "CFO",
    "SEC",
    "FED",
    "ETF",
    "IPO",
    "NASDAQ",
    "NASDAQS",
    "LLC",
    "USA",
    "USD",
    # Existing Common Words (Expanded)
    "AM",
    "HI",
    "AN",
    "AT",
    "BY",
    "DO",
    "GO",
    "HA",
    "HE",
    "IF",
    "IN",
    "IS",
    "IT",
    "ME",
    "MY",
    "NO",
    "ON",
    "OR",
    "SO",
    "TO",
    "UP",
    "US",
    "WE",
    "ALL",
    "ADD",
    "AMP",
    "AND",
    "ANY",
    "ARE",
    "ASK",
    "BAD",
    "BIG",
    "BIT",
    "BOY",
    "BUY",
    "CAN",
    "CAP",
    "DAY",
    "DIP",
    "DOT",
    "DUE",
    "EAT",
    "END",
    "ERA",
    "EYE",
    "FIX",
    "FOR",
    "FUN",
    "GET",
    "HAS",
    "HIT",
    "HOW",
    "INK",
    "ITS",
    "JOB",
    "KEY",
    "LAD",
    "LET",
    "LOW",
    "MAN",
    "MAY",
    "NET",
    "NEW",
    "NOT",
    "NOW",
    "OFF",
    "OIL",
    "OLD",
    "ONE",
    "OUT",
    "OWN",
    "PAY",
    "PUT",
    "RAN",
    "RAW",
    "RUN",
    "SAY",
    "SEE",
    "SHE",
    "SUN",
    "THE",
    "TOP",
    "TRY",
    "TWO",
    "USE",
    "WAY",
    "WHO",
    "WHY",
    "WIN",
    "YES",
    "YOU",
    "ABLE",
    "ALOT",
    "ALSO",
    "AUTO",
    "BACK",
    "BALL",
    "BEAR",
    "BEEN",
    "BEST",
    "BILL",
    "BLUE",
    "BOTH",
    "BULL",
    "CASH",
    "CHIP",
    "CITY",
    "CORE",
    "COST",
    "DARK",
    "DASH",
    "DATA",
    "DEAL",
    "DEMO",
    "DONE",
    "DOWN",
    "DRAW",
    "DROP",
    "EACH",
    "ELSE",
    "EVEN",
    "EVER",
    "EYES",
    "FACE",
    "FACT",
    "FALL",
    "FAST",
    "FEEL",
    "FEW",
    "FILE",
    "FILL",
    "FIND",
    "FINE",
    "FIVE",
    "FLOW",
    "FREE",
    "FROM",
    "FULL",
    "GAIN",
    "GAME",
    "GAVE",
    "GIFT",
    "GIVE",
    "GLAD",
    "GOLD",
    "GOOD",
    "GROW",
    "HALF",
    "HAND",
    "HARD",
    "HEAD",
    "HEAR",
    "HELL",
    "HELP",
    "HERE",
    "HIGH",
    "HOLD",
    "HOME",
    "HOPE",
    "HURT",
    "IDEA",
    "INFO",
    "INTO",
    "JOIN",
    "JUST",
    "KEEP",
    "KICK",
    "KIND",
    "KNOW",
    "LAND",
    "LAST",
    "LATE",
    "LEAD",
    "LEFT",
    "LESS",
    "LIFE",
    "LIKE",
    "LINE",
    "LINK",
    "LIST",
    "LIVE",
    "LONG",
    "LOOK",
    "LOST",
    "LOVE",
    "LUCK",
    "MADE",
    "MAKE",
    "MANY",
    "MARK",
    "MEET",
    "MIND",
    "MISS",
    "MOON",
    "MOST",
    "MOVE",
    "MUCH",
    "MUST",
    "NAME",
    "NEAR",
    "NEXT",
    "NICE",
    "NONE",
    "NOTE",
    "ONTO",
    "OPEN",
    "OVER",
    "PART",
    "PASS",
    "PAST",
    "PEAK",
    "PLAN",
    "PLAY",
    "POST",
    "PUMP",
    "PURE",
    "QUIT",
    "REAL",
    "REST",
    "RIDE",
    "RISE",
    "ROAD",
    "ROCK",
    "ROLL",
    "ROOF",
    "ROOM",
    "SAID",
    "SAME",
    "SAVE",
    "SELF",
    "SELL",
    "SENT",
    "SHIP",
    "SHOP",
    "SHOW",
    "SIDE",
    "SIGN",
    "SITE",
    "SIZE",
    "SLOW",
    "SNOW",
    "SOME",
    "SOON",
    "STAY",
    "STEP",
    "STOP",
    "SUCH",
    "SURE",
    "TACO",
    "TAKE",
    "TALK",
    "TEAM",
    "TECH",
    "TELL",
    "THAN",
    "THAT",
    "THEM",
    "THEN",
    "THEY",
    "THIS",
    "THRU",
    "THUS",
    "TILL",
    "TIME",
    "TOLD",
    "TOOK",
    "TOWN",
    "TRUE",
    "TURN",
    "UNIT",
    "UPON",
    "USED",
    "USER",
    "VERY",
    "VIEW",
    "WAIT",
    "WALK",
    "WANT",
    "WARS",
    "WEEK",
    "WELL",
    "WENT",
    "WERE",
    "WHAT",
    "WHEN",
    "WHOM",
    "WILL",
    "WIND",
    "WISH",
    "WITH",
    "WORD",
    "WORK",
    "YEAR",
    "YELL",
    "YOUR",
    "ZERO",
    # --- Caps acronyms / jargon written in uppercase in WSB posts that collide
    # with obscure ticker symbols (surface here after case-aware matching). ---
    "MIN",  # minute / minimum
    "EDIT",  # "EDIT:" post addendum
    "GPU",
    "GPUS",
    "EPS",  # earnings per share
    "PDT",  # pattern day trader
    "RSI",  # relative strength index
    "RSU",  # restricted stock units
    "WTF",
    "EUV",  # lithography
    "HBM",  # high-bandwidth memory
    "IEEPA",  # trade-law acronym
    "DLA",  # Defense Logistics Agency
    "FX",  # foreign exchange
    "HQ",  # headquarters
    "CLI",  # command-line interface
    "CS",   # computer science
    "TA",   # technical analysis
    "GLP",  # GLP-1 receptor agonist drug class
    "UK",   # United Kingdom (country)
    "PS",   # postscript ("PS: …")
    "XYZ",  # placeholder / example name
    "IMO",  # "in my opinion"
    "WTI",  # West Texas Intermediate crude oil benchmark
    "IP",   # intellectual property
    "TV",   # television
    "IRS",  # Internal Revenue Service
    "AGI",  # artificial general intelligence
    "EMA",  # European Medicines Agency / exponential moving average
    "API",  # application programming interface
    "DC",   # direct current
    "CPA",  # cost per acquisition
    "SG",   # SG&A (selling, general & administrative)
    "III",  # Roman numeral / "Act III"
    "TX",   # Texas (state abbreviation)
    "NBA",  # National Basketball Association
    "ECON", # economics / "Econ 101"
    # --- Financial metrics written in caps that collide with obscure tickers. ---
    "TTM",  # trailing twelve months
    "ARR",  # annual recurring revenue
    "DCF",  # discounted cash flow
    "NAV",  # net asset value
    "PT",   # price target
    "FCF",  # free cash flow
    "ROIC", # return on invested capital
    "DTE",  # days to expiration (options)
    "SMA",  # simple moving average
    # --- Options/market-structure jargon. ---
    "LEAP",  # singular of LEAPs (long-dated options); LEAPS already blocked
    "ATHS",  # "ATHs" (all-time highs plural) via S-strip; ATH already blocked
    "IPOS",  # "IPOs" plural via S-strip; IPO already blocked
    # --- Regex S-strip collisions: plural/abbreviation → obscure ticker. ---
    "CLAS",  # "CLASS" → CLAS via S-strip (class-action posts)
    "MMS",   # "MMs" (market makers) → MMS via S-strip
    "RHS",   # "RHs" (Robinhood plural) → RHS via S-strip
    # --- Common words the AI path emits as "company" and mis-resolves to an
    # obscure ticker (financial->FISI, stock->SYBT, strategic->STRA,
    # tenet->THC, capital->CBNK, azure->AZRE, si->SI). ---
    "FINANCIAL",
    "STOCK",
    "STRATEGIC",
    "TENET",
    "CAPITAL",
    "AZURE",  # Microsoft Azure cloud; base-name resolves to Azure Power Global (AZRE)
    "SI",     # short interest metric; Silvergate Capital (SI) is bankrupt
    # --- Indices are not tradeable tickers per the labeling policy. ---
    "SPX",
    # --- C-suite / business-role abbreviations. ---
    "COO",   # chief operating officer
    "CTO",   # chief technology officer
    "CTOS",  # "CTOs" plural via S-strip
    "MGMT",  # management
    # --- Government / regulatory bodies (not publicly traded). ---
    "FAA",   # Federal Aviation Administration
    "FTC",   # Federal Trade Commission
    "OCC",   # Office of the Comptroller of the Currency
    "PJM",   # PJM Interconnection grid operator
    # --- Financial metrics that collide with obscure tickers. ---
    "ROI",   # return on investment
    "IRR",   # internal rate of return
    "PV",    # present value
    "CTB",   # cost to borrow
    "PMI",   # purchasing managers index / private mortgage insurance
    "BTM",   # behind-the-meter energy term
    "REIT",  # real estate investment trust (category, not the ETF ticker)
    # --- Crypto / digital assets (not US equity exchange securities). ---
    "BTC",   # Bitcoin
    # --- Geographic codes that are not US ticker symbols. ---
    "JP",    # Japan / "JP Morgan" shorthand
    "UAE",   # United Arab Emirates
    "TSE",   # Toronto Stock Exchange code (exchange prefix, not a ticker)
    # --- Technology / engineering jargon. ---
    "ASIC",  # application-specific integrated circuit
    "DRAM",  # dynamic random-access memory
    "QLC",   # quad-level cell NAND flash
    "MIMO",  # multiple-input multiple-output antenna technology
    "BBU",   # baseband unit (telecom infrastructure)
    "SFR",   # sodium fast reactor
    "EBR",   # experimental breeder reactor designation
    "PPA",   # power purchase agreement
    "PBC",   # public benefit corporation
    # --- Internet / WSB slang that collides with obscure tickers. ---
    "TBH",   # to be honest
    "GEMI",  # Gemini (AI product) — engine frequently hallucinates this
    # --- Common English words / phrases that collide with obscure tickers. ---
    "DRUG",  # common noun
    "GOAT",  # "greatest of all time"
    "MATH",  # mathematics
    "BBQ",   # barbecue
    "JUNE",  # full month name (JUN already blocked)
    "TERM",  # common English word
    "BANG",  # slang / meme-stock group acronym
    "SOAR",  # common verb / military unit abbreviation
    "CAPE",  # CAPE ratio / customs portal
    "SP",    # S&P index fragment / "share price" abbreviation
    "WB",    # Warner Bros (legacy ticker; company is now WBD)
    "TILT",  # WSB / poker slang for emotional trading ("went on tilt")
    "EVE",   # common word (evening / "eve of") — company_to_ticker resolves to EVEX
    # --- Bad company-name base-words: common words/names that map to obscure
    # tickers via company_to_ticker, producing wrong resolutions. ---
    "GREEN",     # adjective → GCDT (Green Circle Biotech)
    "GLOBAL",    # adjective → GBLI (Global Indemnity)
    "CHINA",     # country name → CAAS (China Automotive)
    "BITCOIN",   # crypto → BIXI (Bitcoin Infrastructure); BTC already blocked
    "FIDELITY",  # private brokerage → FDBC (Fidelity D&D Bancorp)
    "JEFFERSON", # common name → JCAP (Jefferson Capital)
    "SPECTRUM",  # common noun → SPB (Spectrum Brands)
    "SIMPSON",   # common name → SSD (Simpson Manufacturing)
    "BASEL",     # city name → BMGL (Basel Medical Group)
    "BEYOND",    # common preposition/adverb → BYND (Beyond Meat); "Beyond Meat" still resolves via 2-word key
    # --- Tech/finance abbreviations that collide with real tickers. ---
    "SSD",    # solid-state drive (storage tech) collides with Simpson Manufacturing (SSD)
    "SMR",    # small modular reactor (nuclear energy jargon) collides with NuScale Power (SMR)
    "TAIL",   # tail risk (finance jargon) collides with Cambria Tail Risk ETF (TAIL)
    "UFO",    # cultural usage collides with Procure Space ETF (UFO)
}
