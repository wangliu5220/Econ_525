import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

TRADING_DAYS_PER_YEAR = 252

# =============================================================================
# 1. LOAD DATA
# =============================================================================
ls_df = pd.read_csv("Portfolio/long_short_returns_ff3.csv", parse_dates=["date"])
sharpe_df = pd.read_csv("Portfolio/sharpe_ratios.csv", index_col="strategy")

STRATS = ["ess", "llm_sentiment", "tangibility", "relevance", "composite"]
FACTORS = ["mkt_rf", "smb", "hml"]

# =============================================================================
# 2. FAMA-FRENCH THREE-FACTOR REGRESSIONS
# =============================================================================
reg_results = {}

for s in STRATS:
    sub = ls_df[["date", s, "mkt_rf", "smb", "hml", "rf"]].dropna()

    y = sub[s] - sub["rf"]
    X = sm.add_constant(sub[FACTORS])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    reg_results[s] = {
        "alpha":       model.params["const"],
        "alpha_tstat": model.tvalues["const"],
        "alpha_pval":  model.pvalues["const"],
        "beta_mkt":    model.params["mkt_rf"],
        "beta_smb":    model.params["smb"],
        "beta_hml":    model.params["hml"],
        "r_squared":   model.rsquared,
        "n_obs":       int(model.nobs),
        "model":       model,
    }

reg_df = pd.DataFrame({s: {k: v for k, v in d.items() if k != "model"}
                       for s, d in reg_results.items()}).T
reg_df.index.name = "strategy"

# =============================================================================
# 3. TABLE 2 — PORTFOLIO RETURN SUMMARIES + SHARPE RATIOS
# =============================================================================
summary_rows = []
for s in STRATS:
    sr = ls_df[s].dropna()
    summary_rows.append({
        "strategy":        s,
        "mean_daily_ret":  sr.mean(),
        "std_daily_ret":   sr.std(),
        "annualised_ret":  sr.mean() * TRADING_DAYS_PER_YEAR,
        "annualised_std":  sr.std() * np.sqrt(TRADING_DAYS_PER_YEAR),
        "annualised_sharpe": sharpe_df.loc[s, "annualised_sharpe"] if s in sharpe_df.index else np.nan,
        "n_days":          len(sr),
    })

table2 = pd.DataFrame(summary_rows).set_index("strategy")

# =============================================================================
# 4. TABLE 3 — ALPHA, T-STATS, FACTOR BETAS, R-SQUARED
# =============================================================================
table3 = reg_df[["alpha", "alpha_tstat", "alpha_pval", "beta_mkt", "beta_smb", "beta_hml", "r_squared", "n_obs"]].copy()

# =============================================================================
# 5. ALPHA DIFFERENCE TEST HELPER
# =============================================================================
def alpha_difference_test(strat_a, strat_b):
    """
    Test whether alpha(strat_a) = alpha(strat_b) by running an OLS regression
    on the difference in excess returns against FF3 factors.
    The intercept gives the alpha difference; the t-stat and two-sided p-value
    are reported. For directional hypotheses (strat_a > strat_b), the
    one-sided p-value is half the two-sided p-value when t > 0.
    """
    sub = ls_df[["date", strat_a, strat_b, "mkt_rf", "smb", "hml", "rf"]].dropna()

    y = (sub[strat_a] - sub["rf"]) - (sub[strat_b] - sub["rf"])
    X = sm.add_constant(sub[FACTORS])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    alpha_diff = model.params["const"]
    t_stat     = model.tvalues["const"]
    p_two      = model.pvalues["const"]
    # One-sided p-value: valid only when the observed direction matches the hypothesis
    p_one      = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    return {
        "alpha_diff":  alpha_diff,
        "t_stat":      t_stat,
        "p_two_sided": p_two,
        "p_one_sided": p_one,
    }

# =============================================================================
# 6. HYPOTHESIS TESTS
# =============================================================================
print("\n" + "="*65)
print("HYPOTHESIS TESTS")
print("="*65)

# ---------------------------------------------------------------------------
# H1: LLM Sentiment alpha > ESS alpha (one-sided)
# ---------------------------------------------------------------------------
print("\nH1: LLM Sentiment alpha > ESS alpha")
print("  Individual alphas:")
for s in ["llm_sentiment", "ess"]:
    a = reg_results[s]["alpha"]
    t = reg_results[s]["alpha_tstat"]
    p = reg_results[s]["alpha_pval"]
    print(f"    {s:<18s}  alpha = {a:.6f}  t = {t:.3f}  p (two-sided) = {p:.4f}")

h1 = alpha_difference_test("llm_sentiment", "ess")
print(f"\n  Difference test (LLM Sentiment - ESS):")
print(f"    Alpha difference = {h1['alpha_diff']:.6f}")
print(f"    t-stat           = {h1['t_stat']:.3f}")
print(f"    p-value (two-sided) = {h1['p_two_sided']:.4f}")
print(f"    p-value (one-sided, H1: diff > 0) = {h1['p_one_sided']:.4f}")
sig_h1 = "SUPPORTED" if h1["p_one_sided"] < 0.05 and h1["alpha_diff"] > 0 else "NOT SUPPORTED"
print(f"    H1 at 5% (one-sided): {sig_h1}")

# ---------------------------------------------------------------------------
# H2: Tangibility and Relevance individually significant (two-sided)
# ---------------------------------------------------------------------------
print("\nH2: Tangibility and Relevance individually significant")
for s in ["tangibility", "relevance"]:
    a   = reg_results[s]["alpha"]
    t   = reg_results[s]["alpha_tstat"]
    p   = reg_results[s]["alpha_pval"]
    sig = "SUPPORTED" if p < 0.05 else ("MARGINAL (10%)" if p < 0.10 else "NOT SUPPORTED")
    print(f"  {s:<14s}  alpha = {a:.6f}  t = {t:.3f}  p = {p:.4f}  H2 at 5%: {sig}")

# ---------------------------------------------------------------------------
# H3: Composite alpha > each individual LLM dimension (one-sided)
#     Also report composite vs ESS for completeness
# ---------------------------------------------------------------------------
print("\nH3: Composite alpha > each individual LLM dimension")
comp_alpha = reg_results["composite"]["alpha"]
comp_tstat = reg_results["composite"]["alpha_tstat"]
comp_pval  = reg_results["composite"]["alpha_pval"]
print(f"  Composite alpha = {comp_alpha:.6f}  (t = {comp_tstat:.3f}, p (two-sided) = {comp_pval:.4f})\n")

h3_comparisons = ["llm_sentiment", "tangibility", "relevance"]
all_supported = True
for s in h3_comparisons:
    res = alpha_difference_test("composite", s)
    dim_alpha = reg_results[s]["alpha"]
    supported = res["p_one_sided"] < 0.05 and res["alpha_diff"] > 0
    if not supported:
        all_supported = False
    print(f"  Composite vs {s:<18s}  dim alpha = {dim_alpha:.6f}  "
          f"diff = {res['alpha_diff']:+.6f}  t = {res['t_stat']:.3f}  "
          f"p (one-sided) = {res['p_one_sided']:.4f}  Supported: {'YES' if supported else 'NO'}")

print(f"\n  H3 fully supported (composite > all LLM dims at 5%): {'YES' if all_supported else 'NO'}")

# Composite vs ESS for completeness
print("\n  Additional: Composite vs ESS")
res_ess = alpha_difference_test("composite", "ess")
print(f"  diff = {res_ess['alpha_diff']:+.6f}  t = {res_ess['t_stat']:.3f}  "
      f"p (two-sided) = {res_ess['p_two_sided']:.4f}  p (one-sided) = {res_ess['p_one_sided']:.4f}")

# =============================================================================
# 7. SAVE OUTPUTS
# =============================================================================
table2.to_csv("Portfolio/table2_summary.csv")
table3.to_csv("Portfolio/table3_regressions.csv")

print("\n" + "="*65)
print("TABLE 2 — Portfolio Return Summaries")
print("="*65)
print(table2.round(6).to_string())

print("\n" + "="*65)
print("TABLE 3 — FF3 Regression Results")
print("="*65)
print(table3.round(6).to_string())

print("\nSaved: table2_summary.csv, table3_regressions.csv")

# =============================================================================
# 8. TABLE 4 — ALPHA DIFFERENCE (HYPOTHESIS TEST) REGRESSIONS
# =============================================================================

rows = []

comparisons = [
    ("LLM Sentiment - ESS", "llm_sentiment", "ess"),
    ("Composite - LLM Sentiment", "composite", "llm_sentiment"),
    ("Composite - Tangibility", "composite", "tangibility"),
    ("Composite - Relevance", "composite", "relevance"),
]

for name, a, b in comparisons:
    sub = ls_df[["date", a, b, "mkt_rf", "smb", "hml", "rf"]].dropna()

    # Construct difference in excess returns
    y = (sub[a] - sub["rf"]) - (sub[b] - sub["rf"])
    X = sm.add_constant(sub[FACTORS])

    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    rows.append({
        "comparison": name,
        "alpha_diff_bp": model.params["const"] * 10000,  # convert to basis points
        "t_stat": model.tvalues["const"],
        "p_value_two_sided": model.pvalues["const"],
        "p_value_one_sided": model.pvalues["const"]/2 if model.tvalues["const"] > 0 else 1 - model.pvalues["const"]/2,
        "beta_mkt": model.params["mkt_rf"],
        "beta_smb": model.params["smb"],
        "beta_hml": model.params["hml"],
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs)
    })

table4 = pd.DataFrame(rows).set_index("comparison")

# Save
table4.to_csv("Portfolio/table4_alpha_differences.csv")

print("\n" + "="*65)
print("TABLE 4 — Alpha Difference Regression Results")
print("="*65)
print(table4.round(4).to_string())