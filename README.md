# Stock Recognizer 📈

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->

---
<!-- Adjust the link of the first and second badges to your own repo -->
<p align="center">
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/StephanAkkerman/stock-recognizer/pyversions.yml?label=python%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13&logo=python&style=flat-square">
  <img src="https://img.shields.io/github/license/StephanAkkerman/stock-recognizer.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

## Introduction

A lightweight, hybrid Python library for extracting stock tickers and company names from messy social media text (Reddit, Twitter, etc.).

The goal is to answer **"what stocks is this post talking about?"** — `recognize()` returns a deduplicated **set** of tickers per text. A symbol mentioned once or twenty times yields the same result; the focus is coverage of *which* tickers appear, not counting every occurrence. Accuracy is measured the same way (set-based, deduplicated per document), so a ticker only needs to be caught once to count as found.

## Table of Contents 🗂

- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
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

# Default: use_ai=True, most accurate.
# First run downloads the fine-tuned adapter from the HF Hub (cached after that)
recognizer = StockRecognizer()

text = "$PLAB DD: easy to understand TSMC supplier"
tickers_ai = recognizer.recognize_ai(text)
print(tickers_ai) # ['PLAB', 'TSM']

# Opt out of AI for a lighter, faster, regex + market-data-only path
fast_recognizer = StockRecognizer(use_ai=False)
tickers = fast_recognizer.recognize(text)
print(tickers) # ['PLAB'] (TSMC needs AI mapping)
```

## Model Training 🧠

The GLiNER2 adapter behind `recognize_ai()` — scraping, labeling policy, training, and benchmarking — lives in the sibling [`stock-recognizer-model`](https://github.com/StephanAkkerman/stock-recognizer-model) repo. `StockRecognizer()` (`use_ai=True` by default) fetches the trained adapter automatically from [`StephanAkkerman/stock-recognizer-model`](https://huggingface.co/StephanAkkerman/stock-recognizer-model) on the Hugging Face Hub (pass `adapter_path=` to use a local adapter instead, or `adapter_revision=` to pin a different published version).

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
