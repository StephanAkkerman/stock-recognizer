import re

import financedatabase as fd
from gliner2 import GLiNER2

from .constants import AMBIGUOUS_WORDS, COMPANY_SEEDS, EXCHANGE_BLACKLIST


class StockRecognizer:
    def __init__(self, use_ai=False):
        print("Initializing Market Intelligence v0.1.7...")
        equities = fd.Equities()
        self.us_equities = equities.select(country="United States")

        self.valid_tickers = {
            t
            for t in self.us_equities.index
            if isinstance(t, str) and not any(ext in t for ext in EXCHANGE_BLACKLIST)
        }

        # Build Company Mapper
        self.company_to_ticker = COMPANY_SEEDS.copy()
        for ticker, row in self.us_equities.iterrows():
            if isinstance(row.get("name"), str):
                name = (
                    row["name"]
                    .upper()
                    .replace(",", "")
                    .replace(".", "")
                    .replace("INC", "")
                    .strip()
                )
                base_name = name.split()[0]
                if len(base_name) > 2 and base_name not in self.company_to_ticker:
                    self.company_to_ticker[base_name] = ticker

        # Simplified Regex: Just find blocks of 2-6 letters
        self.ticker_re = re.compile(r"\b[A-Z]{2,6}\b")
        self.cashtag_re = re.compile(r"\$([A-Z]{1,6})\b")

        self.extractor = None
        if use_ai:
            self.extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

    def _clean_token(self, token):
        """Standardizes tokens by removing 'S and plurals."""
        t = token.upper().strip().replace("$", "")
        if t.endswith("'S"):
            t = t[:-2]
        # If it's a long word ending in S, try the singular (e.g., AAPLS -> AAPL)
        if len(t) > 3 and t.endswith("S") and t not in self.valid_tickers:
            if t[:-1] in self.valid_tickers:
                return t[:-1]
        return t

    def recognize(self, text: str) -> list[str]:
        found = set()
        if not text:
            return []

        # 1. Cashtags (Golden Rule)
        for tag in self.cashtag_re.findall(text.upper()):
            clean_tag = self._clean_token(tag)
            # Cashtags are explicit user intent; trust them even if the
            # symbol is absent from the current market snapshot.
            if clean_tag:
                found.add(clean_tag)

        # 2. Plain Text Regex
        if not self._is_mostly_uppercase(text):
            # Match 2-6 uppercase chunks
            for raw_token in self.ticker_re.findall(text.upper()):
                clean_t = self._clean_token(raw_token)
                if clean_t in self.valid_tickers and clean_t not in AMBIGUOUS_WORDS:
                    found.add(clean_t)
        return list(found)

    def _is_mostly_uppercase(self, text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return False
        return sum(1 for c in letters if c.isupper()) / len(letters) > 0.5

    def recognize_ai(self, text: str) -> list[str]:
        if not self.extractor or not text:
            return self.recognize(text)

        # Start with Regex results
        found = set(self.recognize(text))

        try:
            # We treat 'company' and 'ticker' labels as "potential entities"
            result = self.extractor.extract_entities(text, ["company", "ticker"])
        except:
            return list(found)

        entities = result.get("entities", result) if isinstance(result, dict) else {}

        # Flatten all AI entities into one list to resolve
        all_ai_mentions = entities.get("company", []) + entities.get("ticker", [])

        for mention in all_ai_mentions:
            # 1. Clean the mention
            m_clean = self._clean_token(mention)
            if not m_clean or m_clean in AMBIGUOUS_WORDS:
                continue

            # 2. Try to resolve as a direct Ticker
            if m_clean in self.valid_tickers:
                found.add(m_clean)
                continue

            # 3. Try to resolve as a Company Name
            # (Matches "Micron" -> "MU")
            base_name = m_clean.split()[0]
            ticker_map = self.company_to_ticker.get(
                m_clean, self.company_to_ticker.get(base_name)
            )
            if ticker_map:
                found.add(ticker_map)

        return list(found)
