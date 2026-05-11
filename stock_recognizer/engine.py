import re

import financedatabase as fd
from gliner2 import GLiNER2


class StockRecognizer:
    def __init__(self, use_ai=False):
        print("Initializing Market Intelligence...")

        # 1. Load Market Data
        equities = fd.Equities()
        self.us_equities = equities.select(country="United States")
        self.valid_tickers = set(self.us_equities.index)

        # 2. Company Mapping Dictionary (Seed the biggest companies first to prevent overriding)
        self.company_to_ticker = {
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
        }
        for ticker, row in self.us_equities.iterrows():
            if isinstance(row.get("name"), str):
                name = row["name"].upper().replace(",", "").replace(".", "")
                base_name = name.split()[0]
                # Only add if we haven't already mapped it (preserves our seeds)
                if len(base_name) > 2 and base_name not in self.company_to_ticker:
                    self.company_to_ticker[base_name] = ticker

        # 3. Setup Patterns & Full Blacklist
        # Matches Cashtags up to 6 letters, optional trailing 'S
        self.cashtag_re = re.compile(r"\$([A-Z]{1,6}(?:\'S)?)\b")
        # Matches plain text tickers, optional trailing 'S
        self.ticker_re = re.compile(r"\b([A-Z]{1,6}(?:\'S)?)\b")

        self.blacklist = {
            "ITM",
            "ATM",
            "OTM",
            "IV",
            "DD",
            "FD",
            "LEAPS",
            "PTE",
            "WSB",
            "YOLO",
            "FOMO",
            "TLDR",
            "LFG",
            "MOASS",
            "HODL",
            "GUH",
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
            "LLC",
            "USA",
            "USD",
            "A",
            "I",
            "IT",
            "IS",
            "ARE",
            "FOR",
            "ON",
            "AND",
            "THE",
            "HAS",
            "BE",
        }

        # 4. Optional AI initialization
        self.extractor = None
        if use_ai:
            print("Loading GLiNER2 (this may take a moment)...")
            self.extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    def _is_mostly_uppercase(self, text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        return sum(1 for c in letters if c.isupper()) / len(letters) > 0.4

    def recognize(self, text: str) -> list[str]:
        found = set()
        text_upper = text.upper()

        # Pre-clean: Remove common filings like "10-K" so the "K" isn't picked up as Kellogg
        text_clean = re.sub(r"\b\d+-[A-Z]\b", "", text_upper)

        # 1. Check Cashtags ($AAPL, $AAPLS)
        for tag in self.cashtag_re.findall(text_clean):
            if tag.endswith("'S"):
                tag = tag[:-2]

            if tag in self.valid_tickers:
                found.add(tag)
            elif tag.endswith("S") and tag[:-1] in self.valid_tickers:
                found.add(tag[:-1])

        # 2. Check Plain Text
        if not self._is_mostly_uppercase(text):
            tokens = self.ticker_re.findall(text_clean)
            for token in tokens:
                # Clean possessive ('S)
                if token.endswith("'S"):
                    token = token[:-2]

                # Check directly
                if token in self.valid_tickers and token not in self.blacklist:
                    found.add(token)
                # Check plurals (AAPLS -> AAPL)
                elif token.endswith("S") and len(token) > 1:
                    singular = token[:-1]
                    if (
                        singular in self.valid_tickers
                        and singular not in self.blacklist
                    ):
                        found.add(singular)

        return list(found)

    def recognize_ai(self, text: str) -> list[str]:
        if not self.extractor:
            raise RuntimeError("StockRecognizer initialized with use_ai=False")

        found = set(self.recognize(text))

        try:
            result = self.extractor.extract_entities(text, ["company", "ticker"])
        except AttributeError:
            result = self.extractor.predict_entities(text, ["company", "ticker"])

        entities = result.get("entities", result) if isinstance(result, dict) else {}

        # GLiNER fallback formatting check
        if isinstance(result, list):
            for e in result:
                label = e.get("label", "")
                val = e.get("text", "")
                if label == "company":
                    entities.setdefault("company", []).append(val)
                elif label == "ticker":
                    entities.setdefault("ticker", []).append(val)

        # Map AI-found companies to Tickers
        for company in entities.get("company", []):
            name = company.upper().strip().replace(",", "").replace(".", "")
            base_name = name.split()[0] if name else ""
            ticker = self.company_to_ticker.get(base_name)
            if ticker:
                found.add(ticker)

        # Validate AI-found tickers (Guard against Hallucinations)
        text_upper = text.upper()
        for t in entities.get("ticker", []):
            t_clean = t.replace("$", "").upper().strip()

            # 1. Must be a valid US equity. 2. Not in blacklist. 3. Must literally exist in the text!
            if t_clean in self.valid_tickers and t_clean not in self.blacklist:
                if t_clean in text_upper:
                    found.add(t_clean)

        return list(found)
