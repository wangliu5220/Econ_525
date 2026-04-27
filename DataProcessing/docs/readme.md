# Semantic Alpha: LLM Information Extraction vs. Traditional Sentiment in Asset Pricing

ECON 525 — Empirical Research Project

## Overview

Uses the Gemini API to score financial news headlines for tangibility, relevance, and sentiment, then tests whether these LLM signals generate alpha beyond RavenPack's ESS metric across a 30-stock AI-equity universe (2023–2024).

## Setup

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

**API key:** Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

## Pipeline

Run from the project root in order:

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

## Data Files

| File | Description |
|---|---|
| `data/crsp_ai_sec_1.csv` | CRSP daily returns, 30 AI equities, 2023–2024 |
| `data/F-F_Research_Data_Factors_daily.csv` | Fama-French 3-factor daily data |
| `data/control_dataset_ready.csv` | CRSP + RavenPack ESS merged *(pre-generated)* |
| `data/llm_input_prompts.csv` | Deduplicated headlines for LLM scoring *(pre-generated)* |
| `data/llm_scores_headline.csv` | Headline-level LLM scores *(pre-generated)* |
| `data/llm_scores_daily.csv` | Daily-aggregated LLM scores *(pre-generated)* |
| `data/validation_set.csv` | 40 hand-labeled validation headlines |
| `data/validation_set_scored.csv` | Validation set with LLM scores |
