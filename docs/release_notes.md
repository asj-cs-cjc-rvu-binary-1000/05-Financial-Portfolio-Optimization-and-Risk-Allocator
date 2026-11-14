# Phase 7 — Release Notes

**Release Version:** 1.0.0
**Release Date:** 2026-11-14
**Commit:** *(populated after final commit)*
**Branch:** `feature/phase-4`

## Highlights

This is the final, submission-ready release of the **Financial Portfolio Optimization & Risk Allocator** project for **LA2133: Theory and Practice of Linear Algebra**.

## What's Included

### Code (`src/` and `app/`)
- **Phase 1 (Data):** `yfinance` ingestion, cleaning, CSV caching.
- **Phase 2 (Returns):** Simple/log returns, annualization.
- **Phase 3 (Linear Algebra):** Covariance matrix, correlation, eigendecomposition, PSD checks.
- **Phase 4 (Optimization):** Min-variance, max-Sharpe, target-return SLSQP optimizers.
- **Phase 5 (Risk):** Historical VaR, CVaR, maximum drawdown, efficient frontier.
- **Phase 6 (Dashboard):** Interactive Streamlit app with 3 tabs.

### Tests (`tests/`)
- **252 passing tests** across 13 files.
- All phases covered. Runtime: ~5s. No flakiness.

### Documentation (`docs/`)
- Mathematical foundations (Phase 3–5 LaTeX derivations).
- Optimization specification (Q1–Q7).
- Risk analysis specification (Q1–Q8).
- Dashboard specification (Q1–Q8).
- Final audit, test report, E2E verification.
- Reproducibility guide.
- Academic reports (RVU format, IEEE format).

### Screenshots (`docs/screenshots/`)
- 8 PNG artifacts: correlation heatmap, eigen spectrum, efficient frontier, drawdown, VaR histogram, dashboard (optimization, risk, frontier).

## Known Limitations

1. **Covariance estimation:** Uses sample covariance, which is sensitive to noise. Shrinkage (Ledoit-Wolf) is a future extension.
2. **Max-Sharpe non-convexity:** SLSQP is a local solver; in principle, it may find a local optimum. In practice for 7 assets, it consistently finds the global optimum.
3. **No transaction costs:** The optimizers do not penalize turnover.
4. **Data dependency:** The pipeline requires `data/processed/prices_clean_*.csv`. If missing, run `python -m src.data.stock_data`.

## Reproducibility

The project is fully reproducible. See `docs/reproducibility_guide.md` for the step-by-step fresh-clone setup.

## License

MIT License (see `LICENSE`). Academic coursework use.