import pandas as pd
import numpy as np

TRADING_DAYS_PER_YEAR = 252

# =============================================================================
# 1. LOAD DATA
# =============================================================================
crsp = pd.read_csv("DataProcessing\data\control_dataset_ready.csv", parse_dates=["date"])
llm  = pd.read_csv("DataProcessing\data\llm_scores_daily.csv", parse_dates=["Effective_Trading_Date"])

llm = llm.rename(columns={
    "Effective_Trading_Date": "date",
    "mean_tangibility":      "tangibility",
    "mean_relevance":        "relevance",
    "mean_llm_sentiment":    "llm_sentiment",
    "headline_count":        "llm_headline_count",
})

crsp["RET"] = pd.to_numeric(crsp["RET"], errors="coerce")
crsp.dropna(subset=["RET"], inplace=True)

# =============================================================================
# 2. MERGE LLM SCORES INTO CRSP PANEL
# =============================================================================
panel = crsp.merge(llm, on=["TICKER", "date"], how="left")

# =============================================================================
# 3. COMPOSITE SIGNAL
# =============================================================================
# panel["composite_weighted"] = (panel["llm_sentiment"] + 0.5 * panel[["tangibility", "relevance"]].mean(axis=1))
panel["composite"] = panel[["llm_sentiment", "tangibility", "relevance"]].mean(axis=1)
# panel.to_csv("Portfolio/merged_panel.csv", index=False)


# =============================================================================
# 4. NEXT-DAY RETURN
# =============================================================================
panel.sort_values(["TICKER", "date"], inplace=True)
panel["ret_next"] = panel.groupby("TICKER")["RET"].shift(-1)

# =============================================================================
# 5. TERCILE LONG-SHORT PORTFOLIO CONSTRUCTION
# =============================================================================
SIGNALS = {
    "ess":           "avg_ess",
    "llm_sentiment": "llm_sentiment",
    "tangibility":   "tangibility",
    "relevance":     "relevance",
    "composite":     "composite",
    # "composite_weighted": "composite_weighted",
}

def tercile_ls(df, signal_col):
    sub = df.dropna(subset=[signal_col, "ret_next"]).copy()
    if len(sub) < 3:
        return np.nan
    try:
        sub["tercile"] = pd.qcut(sub[signal_col], q=3, labels=[1, 2, 3])
    except ValueError:
        return np.nan
    long_r  = sub.loc[sub["tercile"] == 3, "ret_next"].mean()
    short_r = sub.loc[sub["tercile"] == 1, "ret_next"].mean()
    return long_r - short_r

results = []
for date, day_df in panel.groupby("date"):
    row = {"date": date}

    # ESS sort: exclude stock-days with no news
    ess_sub = day_df[day_df["news_count"] > 0]
    row["ess"] = tercile_ls(ess_sub, "avg_ess")

    # LLM sorts: restrict to stock-days with LLM scores
    llm_sub = day_df[
        day_df[["llm_sentiment", "tangibility", "relevance"]].notna().all(axis=1)
    ]
    for name, col in SIGNALS.items():
        if name == "ess":
            continue
        row[name] = tercile_ls(llm_sub, col)

    results.append(row)

ls_df = pd.DataFrame(results).set_index("date").sort_index()

strat_cols = list(SIGNALS.keys())
ls_df = ls_df.dropna(subset=strat_cols, how="all")

ls_df.to_csv("Portfolio/long_short_returns.csv")

# =============================================================================
# 6. SHARPE RATIOS
# =============================================================================
sharpes = {}
for s in strat_cols:
    sr = ls_df[s].dropna()
    sharpes[s] = (sr.mean() / sr.std()) * np.sqrt(TRADING_DAYS_PER_YEAR) if len(sr) > 1 else np.nan

sharpe_df = pd.DataFrame.from_dict(sharpes, orient="index", columns=["annualised_sharpe"])
sharpe_df.index.name = "strategy"


# =============================================================================
# 7. FAMA-FRENCH FACTOR PLACEHOLDERS
# =============================================================================
for fc in ["mkt_rf", "smb", "hml", "rf"]:
    ls_df[fc] = np.nan

# =============================================================================
# 8. OUTPUT
# =============================================================================
ls_df.to_csv("Portfolio/portfolio_results.csv")
sharpe_df.to_csv("Portfolio/sharpe_ratios.csv")

print(f"Panel: {len(panel):,} stock-days | {panel['date'].nunique()} trading days | {panel['TICKER'].nunique()} tickers")
print(f"Stock-days with LLM scores: {panel['llm_sentiment'].notna().sum():,} / {len(panel):,}")
print(f"Long-short return series: {len(ls_df)} trading days\n")

print("Non-NaN days per strategy:")
for s in strat_cols:
    print(f"  {s:<18s}  {ls_df[s].notna().sum()}")

print("\nAnnualised Sharpe Ratios:")
for s, v in sharpes.items():
    print(f"  {s:<18s}  {v:>8.4f}")

print("\nDaily Long-Short Return Summary:")
print(ls_df[strat_cols].describe().round(6).to_string())


print("\nSaved: portfolio_results.csv, sharpe_ratios.csv")
print("NOTE: mkt_rf, smb, hml, rf are NaN placeholders -- merge FF3 daily data before running alpha regressions.")