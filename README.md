# Semantic Alpha: LLM Information Extraction vs. Traditional Sentiment in Asset Pricing

ECON 525 — Empirical Research Project

## Overview

Uses the Gemini API to score financial news headlines for tangibility, relevance, and sentiment, then tests whether these LLM signals generate alpha beyond RavenPack's ESS metric across a 30-stock AI-equity universe (2023–2024).

## Repository Structure

```
├── DataProcessing/    # Dataset building, LLM scoring, and validation
│   ├── scripts/
│   ├── data/
│   ├── docs/
│   ├── requirements.txt
│   └── .env.example
└── Portfolio/         # Portfolio construction and factor regressions
```

## Setup

**Requirements:** Python 3.10+

```bash
cd DataProcessing
pip install -r requirements.txt
```

**API key:** Create a `.env` file in `DataProcessing/`:

```
GEMINI_API_KEY=your_key_here
```

## Pipeline

Run from `DataProcessing/` in order:

### 1. Build datasets

```bash
python scripts/build_datasets.py
```

Cleans CRSP returns and RavenPack news, aligns headlines to trading dates, outputs:
- `data/control_dataset_ready.csv` — CRSP returns merged with daily ESS (15,244 rows)
- `data/llm_input_prompts.csv` — deduplicated headlines for scoring (26,409 rows)

> **Note:** Requires the raw RavenPack file (~13 GB, not included due to size). Pre-generated outputs are in `data/`.

### 2. Score headlines

```bash
python scripts/llm_score_headlines.py            # full run (~26,000 headlines)
python scripts/llm_score_headlines.py --validate  # validation set only (40 headlines)
```

Calls Gemini in batches of 30 with checkpoint/resume support. Scores each headline on three dimensions (tangibility, relevance, sentiment) on a [-1, 1] scale.

### 3. Validate and generate final outputs

```bash
python scripts/validate_llm_scores.py
```

Runs coverage, null-rate, distribution, and validation-correlation checks, then writes:
- `data/llm_scores_headline.csv` — headline-level scores (26,409 rows)
- `data/llm_scores_daily.csv` — daily-aggregated scores by ticker (8,995 rows)
- `data/score_distributions.png` — score distribution histograms

### 4. Generate Signal Portfolios

```bash
python Portfolio/Portfolio_builder.py
```
Generates trading portfolios for ESS, LLM_Sentiment, Tangibility, Relevance, Composite:
- `Portfolio/long_short_returns.csv` — daily portfolio trading returns (484 rows/trading days)
- `Portfolio/portfolio_results.csv` — daily portfolio trading returns with place holders for Fama-French
- `Portfolio/sharpe_ratios.csv` — annualized sharpe ratios for each signal 

### 5. Obtain Regression

```bash
python Portfolio/create_lsr_ff.py
python Portfolio/regressions.py
```

Runs regressions and generates summary statistics of Fama French
- `Portfolio/table2_summary.csv` - summary of returns
- `Portfolio/table3_regressions.csv` - summary stats of regression (alphas, and fama french betas)
- `Portfolio/F-F_Research_Data_Factors_daily.csv' - daily returns table with fama french filled in
- `Portfolio/table4_alpha_differences.csv` - alpha differences

## Data Files

| File | Description |
|---|---|
| `DataProcessing/data/crsp_ai_sec_1.csv` | CRSP daily returns, 30 AI equities, 2023–2024 |
| `DataProcessing/data/F-F_Research_Data_Factors_daily.csv` | Fama-French 3-factor daily data |
| `DataProcessing/data/control_dataset_ready.csv` | CRSP + RavenPack ESS merged *(pre-generated)* |
| `DataProcessing/data/llm_input_prompts.csv` | Deduplicated headlines for LLM scoring *(pre-generated)* |
| `DataProcessing/data/llm_scores_headline.csv` | Headline-level LLM scores *(pre-generated)* |
| `DataProcessing/data/llm_scores_daily.csv` | Daily-aggregated LLM scores *(pre-generated)* |
| `DataProcessing/data/validation_set.csv` | 40 hand-labeled validation headlines |
| `DataProcessing/data/validation_set_scored.csv` | Validation set with LLM scores |
| `Portfolio/long_short_returns.csv`|daily portfolio trading returns (484 rows/trading days)|
| `Portfolio/portfolio_results.csv`| daily portfolio trading returns with place holders for Fama-French |
| `Portfolio/sharpe_ratios.csv`| annualized sharpe ratios for each signal |
| `Portfolio/table2_summary.csv`| summary of returns |
| `Portfolio/table3_regressions.csv`| summary stats of regression (alphas, and fama french betas) |
| `Portfolio/F-F_Research_Data_Factors_daily.csv`| daily returns table with fama french filled in |

