# Stock Recognizer 📈

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->

---
<!-- Adjust the link of the first and second badges to your own repo -->
<p align="center">
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/StephanAkkerman/stock-recognizer/pyversions.yml?label=python%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13&logo=python&style=flat-square">
  <img src="https://img.shields.io/github/license/StephanAkkerman/stock-recognizer.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

> [!WARNING]
> This project is still in its alpha stage. The API is not stable and may change without warning. Use with caution and expect breaking changes. I am still busy optimizing the engine.

## Introduction

A lightweight, hybrid Python library for extracting stock tickers and company names from messy social media text (Reddit, Twitter, etc.).

## Table of Contents 🗂

- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Label Guidelines](#label-guidelines)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Key Features 🔑

- **Hybrid Engine**: Combines Regex, `financedatabase` (Market Data), and GLiNER2 (AI).
- **Context Aware**: Distinguishes between "DD" (Due Diligence) and "DD" (DuPont).
- **Yelling Protection**: Smart filters for posts written in ALL CAPS.
- **Auto-Mapping**: Automatically converts "Apple" or "TSMC" to `AAPL` and `TSM`.

## Installation ⚙️
<!-- Adjust the link of the second command to your own repo -->

Installation can be done via pip:

```bash
pip install stock-recognizer
```

## Usage ⌨️
```python
from stock_recognizer import StockRecognizer

# Initialize (Market Data only for speed)
recognizer = StockRecognizer(use_ai=False)

text = "$PLAB DD: easy to understand TSMC supplier"
tickers = recognizer.recognize(text)
print(tickers) # ['PLAB'] (TSMC needs AI mapping)

# Initialize with AI for deep extraction
recognizer_ai = StockRecognizer(use_ai=True)
tickers_ai = recognizer_ai.recognize_ai(text)
print(tickers_ai) # ['PLAB', 'TSM']
```

## Label Guidelines 🏷️

When annotating training data in Label Studio, use exactly two labels: `ticker` and `company`. The distinction is based on the **form of the text**, not the author's intent — this makes annotation consistent and removes judgment calls.

### Decision tree

**1. Cashtag (`$` prefix) → always `ticker`**
```
$AMC  $TSLA  $gme  $EUV  $DRAM
```
The `$` signals explicit stock intent. Label as `ticker` unconditionally regardless of whether the symbol exists in any index.

**2. ALL-CAPS, 1–5 characters, resolves to a known ticker → `ticker`**
```
AMC   META   NVDA   SOFI   BP
```
If the all-caps form is a real ticker symbol, use `ticker` even if the author is talking about the company.

**3. ALL-CAPS, but the ticker symbol differs → `company`**
```
NVIDIA  (ticker is NVDA)
TSMC    (ticker is TSM)
APPLE   (ticker is AAPL)
```
When the all-caps abbreviation is a company name and not the actual ticker symbol, use `company`.

**4. Written / mixed-case name → `company`**
```
Meta   Nvidia   Micron   AMC Theatres   Goldman Sachs
```
Human-readable names in any non-all-caps form are company references. The engine resolves them to tickers via its company-name mapping.

**5. Informal lowercase ticker (Reddit shorthand) → `ticker`**
```
gme   amc   tsla   spy
```
Lowercase versions of ticker symbols used as shorthand (common on Reddit) should be labeled `ticker`, not `company`.

### What not to label

Skip these — they cannot resolve to a tradeable ticker:

| Category | Examples |
|---|---|
| Government / regulatory bodies | `CSRC`, `SEC`, `NASA`, `FINRA`, `NDAA` |
| Financial metric acronyms | `PDT`, `EV` (enterprise value), `SG&A`, `RSUs`, `PT` (price target), `ATM`, `IV` |
| Memory / chip technology terms | `DRAM`, `HBM`, `EUV`, `NAND` (without `$` prefix) |
| Media outlets with no public stock | `CNBC`, `Bloomberg`, `HBO`, `MSNBC` |

### Quick reference

| Text in post | Label | Reason |
|---|---|---|
| `$AMC`, `$gme`, `$EUV` | `ticker` | Cashtag — unconditional |
| `AMC`, `SNAP`, `SOFI` | `ticker` | All-caps ticker symbol |
| `gme`, `amc`, `tsla` | `ticker` | Informal lowercase shorthand |
| `NVIDIA`, `TSMC`, `APPLE` | `company` | All-caps name, ticker differs |
| `Meta`, `Nvidia`, `Micron` | `company` | Written name form |
| `AMC Theatres`, `Goldman Sachs` | `company` | Multi-word name |
| `CSRC`, `PDT`, `EV`, `DRAM` | *(skip)* | Non-tradeable entity |

### Keeping data consistent

Run the policy fixer after annotating to catch labeling errors automatically:

```bash
python trainer/fix_label_policy.py          # preview changes
python trainer/fix_label_policy.py --apply  # apply fixes
```

After fixing `data/labeled/`, regenerate `data/augmented/` before retraining.

---

## Citation ✍️
<!-- Be sure to adjust everything here so it matches your name and repo -->
If you use this project in your research, please cite as follows:

```bibtex
@misc{project_name,
  author  = {Stephan Akkerman},
  title   = {Stock Recognizer},
  year    = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/stock-recognizer}}
}
```

## Contributing 🛠
<!-- Be sure to adjust the repo name here for both the URL and GitHub link -->
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.\
![https://github.com/StephanAkkerman/stock-recognizer/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/stock-recognizer)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
