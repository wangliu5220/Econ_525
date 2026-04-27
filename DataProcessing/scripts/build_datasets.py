"""
Pipeline to clean, align, and merge CRSP daily returns with RavenPack news
analytics for an empirical asset-pricing study.

Outputs (written to ../data/):
    control_dataset_ready.csv   – CRSP returns LEFT-joined with daily-aggregated ESS
    llm_input_prompts.csv       – Deduplicated headlines/snippets mapped to trading dates
"""

import pathlib
import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

CRSP_PATH = DATA_DIR / "crsp_ai_sec_1.csv"
RP_PATH = DATA_DIR / "ravenpack_ai_sec_1.csv"

OUT_CONTROL = DATA_DIR / "control_dataset_ready.csv"
OUT_LLM = DATA_DIR / "llm_input_prompts.csv"

# Config
RP_CHUNK_SIZE = 500_000          # rows per chunk when streaming RavenPack
RELEVANCE_FLOOR = 75             # minimum event_relevance to keep
NEUTRAL_ESS = 0                  # fill for trading days with no news (ESS is on -1 to 1 scale)
MARKET_CLOSE_ET = 16             # 4 PM Eastern (hour boundary)

# CRSP cleaning — loaded first, needed to build the trading calendar
print("Loading CRSP data...")

crsp = pd.read_csv(CRSP_PATH)
crsp.columns = crsp.columns.str.strip()

# Standardise date
crsp["date"] = pd.to_datetime(crsp["date"], format="mixed")

# Coerce RET to numeric (CRSP encodes missing as 'B', 'C', etc.)
crsp["RET"] = pd.to_numeric(crsp["RET"], errors="coerce")
crsp.dropna(subset=["RET"], inplace=True)

# Ensure PRC and VOL are numeric
crsp["PRC"] = pd.to_numeric(crsp["PRC"], errors="coerce").abs()   # CRSP uses negative PRC for bid/ask avg
crsp["VOL"] = pd.to_numeric(crsp["VOL"], errors="coerce")
crsp["SHROUT"] = pd.to_numeric(crsp["SHROUT"], errors="coerce")

# Uppercase ticker for merge key
crsp["TICKER"] = crsp["TICKER"].astype(str).str.strip().str.upper()

# Restrict to the 30 target AI-exposed equities (drops spurious AAIC, METV, etc.)
AI_TICKERS = {
    "AAPL", "ADBE", "AI", "AMD", "AMZN", "ARM", "ASML", "AVGO", "CEG", "CRM",
    "CRWD", "DDOG", "DLR", "EQIX", "GOOGL", "IBM", "INTC", "META", "MSFT", "MU",
    "NOW", "NVDA", "ORCL", "PLTR", "QCOM", "SMCI", "SNOW", "TLN", "TSM", "VRT",
}
crsp = crsp[crsp["TICKER"].isin(AI_TICKERS)].copy()

print(f"  CRSP rows after cleaning: {len(crsp):,}")

# Trading-day calendar built from CRSP (handles weekends and holidays)
trading_days = np.sort(crsp["date"].unique())
trading_days_set = set(trading_days)

_NAT = np.datetime64("NaT")

def next_trading_day(dt):
    """Return the next trading day strictly after *dt* (numpy datetime64)."""
    candidate = dt + np.timedelta64(1, "D")
    for _ in range(10):
        if candidate in trading_days_set:
            return candidate
        candidate += np.timedelta64(1, "D")
    return _NAT

# Pre-compute two lookups covering the full 2023-2024 window + buffer:
#   _same_or_next[d] → d if d is a trading day, else next trading day
#   _strictly_next[d] → next trading day after d (for after-hours on trading days)
cal_start = trading_days[0] - np.timedelta64(3, "D")
cal_end = trading_days[-1] + np.timedelta64(5, "D")
all_calendar_dates = pd.date_range(cal_start, cal_end, freq="D")

_same_or_next: dict[np.datetime64, np.datetime64] = {}
_strictly_next: dict[np.datetime64, np.datetime64] = {}
for d in all_calendar_dates:
    d64 = d.to_datetime64()
    if d64 in trading_days_set:
        _same_or_next[d64] = d64
    else:
        _same_or_next[d64] = next_trading_day(d64)
    _strictly_next[d64] = next_trading_day(d64)

def effective_trading_date(ts_eastern: pd.Series) -> pd.Series:
    """
    Given a Series of tz-aware Eastern timestamps, return the effective
    trading date (numpy datetime64, tz-naive) respecting the 4 PM cut-off.

    Rules
    -----
    • Before 4 PM ET on a trading day  →  that day
    • At/after 4 PM ET on a trading day  →  next trading day
    • On a non-trading day (any time)  →  next trading day
    """
    date_part = ts_eastern.dt.normalize().dt.tz_localize(None).values
    hour_part = ts_eastern.dt.hour.values

    result = np.empty(len(ts_eastern), dtype="datetime64[ns]")

    for i in range(len(ts_eastern)):
        d = date_part[i]
        if d in trading_days_set and hour_part[i] < MARKET_CLOSE_ET:
            result[i] = d
        elif d in trading_days_set:
            # After hours on a trading day → strictly next trading day
            result[i] = _strictly_next.get(d, _NAT)
        else:
            # Weekend / holiday → next trading day
            result[i] = _same_or_next.get(d, _NAT)

    return pd.Series(result, index=ts_eastern.index, dtype="datetime64[ns]")

# RavenPack: clean, align to trading dates, aggregate (streamed in chunks)
print("Streaming RavenPack data...")

agg_pieces: list[pd.DataFrame] = []      # for control dataset (aggregated)
llm_pieces: list[pd.DataFrame] = []      # for LLM input (row-level)

chunks_processed = 0

for chunk in pd.read_csv(RP_PATH, chunksize=RP_CHUNK_SIZE, low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    chunks_processed += 1

    # ---- Cleaning ----

    # Drop rows without a scored event (sentiment, relevance, similarity key all NaN together)
    chunk.dropna(subset=["event_sentiment_score", "event_relevance"], inplace=True)

    if chunk.empty:
        continue

    # Relevance filter
    chunk["event_relevance"] = pd.to_numeric(chunk["event_relevance"], errors="coerce")
    chunk = chunk[chunk["event_relevance"] >= RELEVANCE_FLOOR].copy()

    if chunk.empty:
        continue

    # Deduplicate on event_similarity_key (keep first occurrence)
    chunk.drop_duplicates(subset=["event_similarity_key"], keep="first", inplace=True)

    # Drop rows missing headline or sentiment
    chunk.dropna(subset=["headline", "event_sentiment_score"], inplace=True)

    if chunk.empty:
        continue

    # Standardise ticker
    chunk["TICKER"] = chunk["ticker"].astype(str).str.strip().str.upper()

    # Ensure sentiment is numeric
    chunk["event_sentiment_score"] = pd.to_numeric(chunk["event_sentiment_score"], errors="coerce")

    # ---- Timezone Alignment ----
    chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], format="mixed", utc=True)
    chunk["ts_eastern"] = chunk["timestamp_utc"].dt.tz_convert("America/New_York")
    chunk["Effective_Trading_Date"] = effective_trading_date(chunk["ts_eastern"])

    # Drop rows where we couldn't resolve a trading date
    chunk.dropna(subset=["Effective_Trading_Date"], inplace=True)

    if chunk.empty:
        continue

    # ---- Collect row-level data for LLM output ----
    llm_pieces.append(
        chunk[["Effective_Trading_Date", "TICKER", "headline", "event_text"]].copy()
    )

    # ---- Aggregate per ticker-day ----
    daily = (
        chunk
        .groupby(["TICKER", "Effective_Trading_Date"], as_index=False)
        .agg(
            avg_ess=("event_sentiment_score", "mean"),
            news_count=("event_sentiment_score", "size"),
        )
    )
    agg_pieces.append(daily)

    if chunks_processed % 20 == 0:
        print(f"  … processed {chunks_processed} chunks ({chunks_processed * RP_CHUNK_SIZE / 1e6:.1f}M rows read)")

print(f"  … finished. Total chunks: {chunks_processed}")

# Combine chunk aggregates and re-aggregate across chunk boundaries
if not agg_pieces:
    print("  WARNING: No RavenPack rows survived filtering. Outputs will have no sentiment data.")
    rp_agg = pd.DataFrame(columns=["TICKER", "Effective_Trading_Date", "avg_ess", "news_count"])
else:
    rp_agg = pd.concat(agg_pieces, ignore_index=True)
    # Weighted average across chunks: total_sentiment / total_count
    rp_agg["_ess_sum"] = rp_agg["avg_ess"] * rp_agg["news_count"]
    rp_agg = (
        rp_agg
        .groupby(["TICKER", "Effective_Trading_Date"], as_index=False)
        .agg(ess_sum=("_ess_sum", "sum"), news_count=("news_count", "sum"))
    )
    rp_agg["avg_ess"] = rp_agg["ess_sum"] / rp_agg["news_count"]
    rp_agg.drop(columns=["ess_sum"], inplace=True)

print(f"  Aggregated RavenPack: {len(rp_agg):,} ticker-days with news")

# Merge CRSP with RavenPack (left join)
print("Merging CRSP with RavenPack...")

merged = crsp.merge(
    rp_agg,
    how="left",
    left_on=["TICKER", "date"],
    right_on=["TICKER", "Effective_Trading_Date"],
)

# Fill missing sentiment with neutral score
merged["avg_ess"] = merged["avg_ess"].fillna(NEUTRAL_ESS)
merged["news_count"] = merged["news_count"].fillna(0).astype(int)

# Drop the redundant join key
merged.drop(columns=["Effective_Trading_Date"], inplace=True, errors="ignore")

print(f"  Merged dataset rows: {len(merged):,}")

# Save outputs
print("Writing output files...")

# Output 1 – Control dataset
merged.sort_values(["date", "TICKER"], inplace=True)
merged.to_csv(OUT_CONTROL, index=False)
print(f"  ✓ {OUT_CONTROL.name}  ({len(merged):,} rows)")

# Output 2 – LLM input (deduplicated headlines + snippets)
if llm_pieces:
    llm_df = pd.concat(llm_pieces, ignore_index=True)
    llm_df.drop_duplicates(
        subset=["Effective_Trading_Date", "TICKER", "headline"],
        keep="first",
        inplace=True,
    )
    llm_df.sort_values(["Effective_Trading_Date", "TICKER"], inplace=True)
else:
    llm_df = pd.DataFrame(columns=["Effective_Trading_Date", "TICKER", "headline", "event_text"])
llm_df.to_csv(OUT_LLM, index=False)
print(f"  ✓ {OUT_LLM.name}  ({len(llm_df):,} rows)")

print("\nDone.")
