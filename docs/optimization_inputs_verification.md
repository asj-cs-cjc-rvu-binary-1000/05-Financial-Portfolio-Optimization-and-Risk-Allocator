# Phase 4: Optimization Inputs Verification

**Date:** 2026-11-13
**Status:** Verified

## 1. Repository State
- **Branch:** `feature/phase-4`
- **Verified Commit Hash:** `5a963cd85349436a7f5da225c804ffbd2a37d0e5`
- **Commit Subject:** `chore: document plotly 6.3 + kaleido 0.2.1 downgrade rationale and tabulate usage`
- **Git history:** Fully unshallowed — full history visible from `06e56d2 Initial commit` through `5a963cd`. No `(grafted)` marker.
- **Working tree:** Three intentional changes pending commit for this task:
  - `M .devcontainer/devcontainer.json` (fix invalid image tag)
  - `M requirements.txt` (remove duplicate `tabulate==0.9.0`)
  - `?? docs/optimization_inputs_verification.md` (this file)
- **Data CSVs:** `data/raw/` and `data/processed/` track only `.gitkeep`; actual CSVs are ignored.

## 2. Environment Verification
- **Python:** 3.14.10 at `/usr/local/bin/python`
- **pytest:** 9.2.0
- **Dependencies:** All pinned packages from `requirements.txt` installed successfully.

## 3. Optimization Inputs Structure
The structures passed to the Phase 4 optimization engine are confirmed mutually consistent in shape and labeling.

### Expected Returns Vector (`mu`)
- **Source:** `src/preprocessing/returns.py` → `process_returns_pipeline`
- **Type:** `pandas.Series`
- **Index:** Ticker symbols (e.g., `'AAPL'`, `'MSFT'`)
- **Values:** Annualized expected returns (daily mean × 252)
- **Convention:** Arithmetic (simple) returns

### Covariance Matrix (`Sigma`)
- **Source:** `src/linear_algebra/covariance.py` → `compute_covariance_matrix(..., annualize=True)`
- **Type:** `pandas.DataFrame`
- **Index & Columns:** Ticker symbols, matching the index of `mu`
- **Values:** Annualized covariance matrix
- **Formula:** `X_centered.T @ X_centered / (T - 1)`, scaled by 252
- **Structural check:** Verified symmetric and positive semi-definite (PSD) in Phase 3

### Asset Ordering Invariant
- Ticker order in `mu` is identical to the columns of the daily returns matrix and to the rows and columns of `Sigma`.
- The optimization engine will treat `w` as a length-N vector or `pd.Series` with the same ticker order, and assert label consistency at function boundaries.

## 4. Annualization Convention
- **Trading Days per Year:** 252
- **Returns:** annualized mean = daily mean × 252
- **Covariance:** annualized cov = daily cov × 252
- **Volatility:** annualized std = daily std × √252

## 5. Test Suite Baseline
- Command: `pytest tests/`
- Result: **37 passed in 1.95s** — matches the Phase 3 sign-off exactly, confirming Phase 1–3 analytical modules are stable.

## 6. Environment Fix Note
The Codespace initially failed to build because `.devcontainer/devcontainer.json` referenced a non-existent image tag `mcr.microsoft.com/devcontainers/python:3.14-bullseye`. The tag was corrected to `mcr.microsoft.com/devcontainers/python:3.14-bookworm`, a valid published tag. After a full rebuild, the environment provisioned correctly and reproduced the 37-test baseline.