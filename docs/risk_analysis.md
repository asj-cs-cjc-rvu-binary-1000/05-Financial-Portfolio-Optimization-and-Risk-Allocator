# Phase 5 — Risk Analysis Summary

**Live run** — `python -m src.risk.comparison`

- Data file: `data/processed/prices_clean_AAPL_MSFT_JPM_JNJ_PG_plus2_2018-01-01_today.csv`
- Assets: AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN
- Risk-free rate: 0.02 (annualized decimal, 2%/yr)
- Confidence level: 95%
- Horizon: daily (no √252 scaling)

## Comparison Table

|               |   return |   volatility |   sharpe |   var_95 |   cvar_95 |   max_drawdown | success   |   AAPL |   MSFT |    JPM |    JNJ |     PG |    XOM |   AMZN |
|---------------|----------|--------------|----------|----------|-----------|----------------|-----------|--------|--------|--------|--------|--------|--------|--------|
| EqualWeight   |   0.1988 |       0.1843 |   0.9701 |   0.0166 |    0.0277 |        -0.3019 | True      | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.1429 | 0.1429 |
| MinVolatility |   0.1479 |       0.1587 |   0.8054 |   0.0139 |    0.0231 |        -0.2748 | True      | 0      | 0.0395 | 0.0284 | 0.352  | 0.3383 | 0.1364 | 0.1055 |
| MaxSharpe     |   0.2236 |       0.2018 |   1.0088 |   0.0184 |    0.0299 |        -0.3142 | True      | 0.3378 | 0.2242 | 0.0825 | 0.2075 | 0.0296 | 0.1055 | 0.013  |

## Interpretation

- **Lowest volatility:** `MinVolatility` (σ_p = 0.1587).
- **Highest Sharpe:** `MaxSharpe` (S_p = 1.0088).
- **Lowest VaR95:** `MinVolatility` (VaR = 1.39%, CVaR = 2.31%).
- **Shallowest max drawdown:** `MinVolatility` (MDD = -27.48%).

**Reading the table:** VaR and CVaR are reported as positive loss magnitudes (industry convention). MDD is a signed peak-to-trough cumulative return (≤ 0), consistent with `drawdown_series`. A portfolio with the lowest volatility does not necessarily have the lowest VaR — VaR depends on the tail shape, not just the second moment. If MinVolatility does not have the smallest VaR on this data, that is a legitimate empirical result, not a bug.
