import logging
import os
import re

import financedatabase as fd
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download

from .constants import (
    AMBIGUOUS_WORDS,
    COMPANY_SEEDS,
    EXCHANGE_BLACKLIST,
    GLOBAL_MAJOR_EXCHANGES,
    US_MAJOR_EXCHANGES,
)

# Published LoRA adapter that recognize_ai() loads by default. Trained in the
# sibling stock-recognizer-model repo; see its utils/hf/push_model_to_hf.py.
# Pinned to a tag rather than "main" — main's adapter_config.json has drifted
# from the tagged checkpoints before (a task_type mismatch that breaks
# PeftModel.from_pretrained), so tags are the only reproducible reference.
DEFAULT_ADAPTER_REPO = "StephanAkkerman/stock-recognizer-model"
DEFAULT_ADAPTER_REVISION = "v18"


class StockRecognizer:

    def __init__(
        self,
        use_ai=True,
        include_global_majors=False,
        adapter_path=None,
        adapter_revision=DEFAULT_ADAPTER_REVISION,
    ):
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
                t
                for t in etf_index
                if isinstance(t, str)
                and not any(ext in t for ext in EXCHANGE_BLACKLIST)
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
                words = name.split()
                base_name = words[0]
                if len(base_name) > 2 and base_name not in self.company_to_ticker:
                    self.company_to_ticker[base_name] = ticker
                # Two-word prefix key so "American Express" → AXP without
                # falling back to base_name "AMERICAN" → AAL.
                if len(words) >= 3:
                    two_word = f"{words[0]} {words[1]}"
                    if two_word not in self.company_to_ticker:
                        self.company_to_ticker[two_word] = ticker

        # Simplified Regex: Just find blocks of 2-6 letters
        # Match an UPPERCASE 2-6 letter core, optionally followed by a
        # plural/possessive 's' ("AAPLs", "MSFT's"). Crucially this runs against
        # the ORIGINAL text (not an uppercased copy) so lowercase prose words
        # ("don't", "edit", "away") are NOT matched — only ticker-style caps.
        self.ticker_re = re.compile(r"\b[A-Z]{2,6}(?:['’]?[sS])?\b")
        self.cashtag_re = re.compile(r"\$([A-Z]{1,6})\b")
        self.logger = logging.getLogger(__name__)

        self.extractor = None
        self.use_ai = use_ai
        if use_ai:
            # 1. Load the Large base model
            self.extractor = GLiNER2.from_pretrained("fastino/gliner2-large-v1")

            # 2. Snap on the fine-tuned adapter: use a local path if given,
            # otherwise fetch the published one from the HF Hub (cached
            # locally by huggingface_hub after the first download).
            resolved_adapter_path = adapter_path
            if not resolved_adapter_path:
                try:
                    resolved_adapter_path = snapshot_download(
                        repo_id=DEFAULT_ADAPTER_REPO, revision=adapter_revision
                    )
                except Exception:
                    self.logger.warning(
                        f"Could not download adapter from {DEFAULT_ADAPTER_REPO} "
                        f"(revision={adapter_revision}); using the base model without "
                        "fine-tuning."
                    )
                    resolved_adapter_path = None

            if resolved_adapter_path and os.path.exists(resolved_adapter_path):
                self.logger.info(
                    f"Loading LoRA adapter from {resolved_adapter_path}..."
                )
                self.extractor.load_adapter(resolved_adapter_path)

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
        upper_text = text.upper()
        for m in self.cashtag_re.finditer(upper_text):
            # Skip exchange-prefix format: $NASDAQ:CSCO → NASDAQ is the exchange,
            # not the ticker. CSCO will be caught by the ticker_re pass below.
            if upper_text[m.end() : m.end() + 1] == ":":
                continue
            tag = m.group(1)
            clean_tag = self._clean_token(tag)
            # Cashtags are explicit user intent; trust them even if the
            # symbol is absent from the current market snapshot.
            if clean_tag:
                found.add(clean_tag)

        # 2. Plain Text Regex — match already-uppercase ticker-style tokens in
        # the original text (case carries the signal: "AMC" is a ticker, "amc"
        # / "away" / "don't" are prose). Skip dot-suffix fragments like .SA/.KL.
        if not self._is_mostly_uppercase(text):
            for match in self.ticker_re.finditer(text):
                start, end = match.span()
                if start > 0 and text[start - 1] == ".":
                    continue
                if end < len(text) and text[end] == ".":
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
            m
            for m in entities.get("company", [])
            if _all_caps_re.match(str(m))
            and str(m).upper().replace("$", "") in self.valid_tickers
        ]
        company_entities = [m for m in entities.get("company", []) if m not in promoted]
        ticker_entities = list(entities.get("ticker", [])) + promoted
        all_ai_mentions = company_entities + ticker_entities

        # Pre-compute once for the surface-form guard below.
        text_upper = text.upper()

        for mention in all_ai_mentions:
            # Surface-form guard: GLiNER2 does span detection, so every
            # legitimate extraction must appear verbatim in the document.
            # If it doesn't, the model hallucinated the entity and we drop it.
            if str(mention).upper() not in text_upper:
                continue

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
            # (Matches "Micron" -> "MU", "Micron Technology" -> "MU")
            words_list = m_clean.split()
            if len(words_list) > 1:
                # Multi-word: exact match first, then 2-word prefix.
                # Never fall back to a bare first-word key — that would
                # let "American Express" resolve via "AMERICAN" → AAL.
                two_word = f"{words_list[0]} {words_list[1]}"
                ticker_map = self.company_to_ticker.get(
                    m_clean
                ) or self.company_to_ticker.get(two_word)
            else:
                ticker_map = self.company_to_ticker.get(m_clean)
            if ticker_map and ticker_map not in AMBIGUOUS_WORDS:
                found.add(ticker_map)

        return list(found)
