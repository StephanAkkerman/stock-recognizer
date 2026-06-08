import logging
import os
import re

import financedatabase as fd
from gliner2 import GLiNER2

from .constants import (
    AMBIGUOUS_WORDS,
    COMPANY_SEEDS,
    EXCHANGE_BLACKLIST,
    GLOBAL_MAJOR_EXCHANGES,
    US_MAJOR_EXCHANGES,
)


class StockRecognizer:

    def __init__(self, use_ai=False, include_global_majors=False, adapter_path=None):
        print("Initializing Market Intelligence v0.1.7...")
        equities = fd.Equities()

        exchanges = set(US_MAJOR_EXCHANGES)
        if include_global_majors:
            exchanges.update(GLOBAL_MAJOR_EXCHANGES)
        self.market_equities = equities.select(exchange=list(exchanges))

        # ETFs are absent from fd.Equities() but heavily mentioned on Reddit
        # (SPY, QQQ, VOO, IWM, JETS, ULCC...). Without them in valid_tickers,
        # `recognize_ai` silently drops correct AI predictions of these symbols.
        # They're kept out of `company_to_ticker` below — ETF legal names like
        # "SPDR S&P 500 ETF Trust" don't make useful company aliases.
        try:
            etf_index = fd.ETFs().select().index
            etf_tickers = {
                t for t in etf_index
                if isinstance(t, str) and not any(ext in t for ext in EXCHANGE_BLACKLIST)
            }
        except Exception:
            etf_tickers = set()

        self.ambiguous = AMBIGUOUS_WORDS
        self.valid_tickers = {
            t
            for t in self.market_equities.index
            if isinstance(t, str) and not any(ext in t for ext in EXCHANGE_BLACKLIST)
        } | etf_tickers

        # Build Company Mapper
        self.company_to_ticker = COMPANY_SEEDS.copy()
        for ticker, row in self.market_equities.iterrows():
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
        self.logger = logging.getLogger(__name__)

        self.extractor = None
        self.use_ai = use_ai
        if use_ai:
            # 1. Load the Large base model
            self.extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

            # 2. Snap on your custom adapter
            # Use the provided path, fallback if none provided
            if adapter_path and os.path.exists(adapter_path):
                self.extractor.load_adapter(adapter_path)

            if adapter_path and os.path.exists(adapter_path):
                self.logger.info(f"Loading LoRA adapter from {adapter_path}...")
                self.extractor.load_adapter(adapter_path)

            # 3. Store the label descriptions — must match ENTITY_DESCRIPTIONS in train.py
            self.ai_labels = {
                "ticker": "A stock market ticker symbol, usually 1-5 letters, often preceded by a dollar sign (e.g., $AAPL, TSLA). MUST NOT be option strikes, prices, index names, or internet slang acronyms.",
                "company": "The name of a corporation, hedge fund, or business entity. MUST NOT be an uppercase ticker symbol, an index, or generic finance terms.",
            }

    def get_ai_entities(self, text):
        """Helper to get flattened AI results."""
        raw = self.extractor.extract_entities(
            text, self.ai_labels, threshold=0.7, include_spans=True
        )
        flat = []
        if isinstance(raw, dict) and "entities" in raw:
            for label, items in raw["entities"].items():
                for item in items:
                    flat.append(
                        {"start": item["start"], "end": item["end"], "label": label}
                    )
        return flat

    def _clean_token(self, token):
        """Standardizes tokens by removing 'S and plurals."""
        t = token.upper().strip().replace("$", "")
        if t.endswith("'S") or t.endswith("’S"):
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
            text_upper = text.upper()
            # Match 2-6 uppercase chunks, but skip dot-suffix fragments like .SA/.KL
            for match in self.ticker_re.finditer(text_upper):
                start, end = match.span()
                if start > 0 and text_upper[start - 1] == ".":
                    continue
                if end < len(text_upper) and text_upper[end] == ".":
                    continue
                raw_token = match.group(0)
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
            # Pass label description dicts so inference prompt matches training
            result = self.extractor.extract_entities(text, self.ai_labels)
        except Exception:
            self.logger.warning("Failed to extract entities with AI model.")
            return list(found)

        entities = result.get("entities", result) if isinstance(result, dict) else {}

        # Relabeling guard: all-caps tokens in company results that are valid tickers
        # should be treated as tickers (model occasionally misclassifies ticker-shaped
        # tokens as company when context is ambiguous).
        _all_caps_re = re.compile(r"^[A-Z][A-Z0-9]{0,5}$")
        promoted = [
            m for m in entities.get("company", [])
            if _all_caps_re.match(str(m)) and str(m).upper().replace("$", "") in self.valid_tickers
        ]
        company_entities = [m for m in entities.get("company", []) if m not in promoted]
        ticker_entities = list(entities.get("ticker", [])) + promoted
        all_ai_mentions = company_entities + ticker_entities

        for mention in all_ai_mentions:
            # 1. Clean the mention
            m_clean = self._clean_token(mention)
            if any(ext in m_clean for ext in EXCHANGE_BLACKLIST):
                continue
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
