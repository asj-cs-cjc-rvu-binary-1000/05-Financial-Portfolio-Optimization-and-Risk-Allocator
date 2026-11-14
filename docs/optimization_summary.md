# Phase 4 — Portfolio Optimization Summary

**Live run** — `python -m src.optimization.optimization_pipeline`

- Data file: `data/processed/prices_clean_AAPL_MSFT_JPM_JNJ_PG_plus2_2018-01-01_today.csv`
- Assets: AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN
- Risk-free rate: 0.02 (annualized decimal, 2%/yr)

## Comparison Table

|               |   return |   volatility |   variance |   sharpe | success   |   n_iter |   AAPL |   MSFT |    JPM |    JNJ |     PG |    XOM |   AMZN |
|---------------|----------|--------------|------------|----------|-----------|----------|--------|--------|--------|--------|--------|--------|--------|
| EqualWeight   |   0.1988 |       0.1843 |     0.034  |   0.9701 | True      |        0 | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.1429 |
| MinVolatility |   0.1479 |       0.1587 |     0.0252 |   0.8054 | True      |       25 | 0      | 0.0395 | 0.0284 | 0.352  | 0.3383 | 0.1364 | 0.1055 |
| MaxSharpe     |   0.2236 |       0.2018 |     0.0407 |   1.0088 | True      |       10 | 0.3378 | 0.2242 | 0.0825 | 0.2075 | 0.0296 | 0.1055 | 0.013  |

## Interpretation

- **Highest Sharpe ratio:** `MaxSharpe` with Sharpe = 1.0088.
- **Lowest volatility:** `MinVolatility` with σ_p = 0.1587.
- **Max-Sharpe concentration:** 5 of 7 assets hold weight > 5% (max-Sharpe typically concentrates; equal-weight never does).
- **Interpretation:** the min-variance row has the smallest volatility by construction; the max-Sharpe row has the largest Sharpe by construction. The equal-weight row is the naive benchmark — if an optimizer does not beat it on its own metric, investigate rather than assume a bug (spec Section 8: SLSQP is a local solver and the max-Sharpe problem is non-convex).
