# Team Ownership Plan – Phase 0-8

**Repository:** 05-Financial-Portfolio-Optimization-and-Risk-Allocator  
**Python:** 3.14.2 (verified)  
**Team:** 2 members – shared ownership with primary responsibilities

## Ownership Matrix

| Phase / Folder | Primary Owner | Support | Notes |
|----------------|---------------|---------|-------|
| Phase 0: Setup (0.1-0.9) | Both | Both | Skeleton, requirements, devcontainer, README, LICENSE, env verification – completed together |
| `src/data/` | Member A | Member B | yfinance download, data loading |
| `src/preprocessing/` | Member A | Member B | Returns calculation, cleaning |
| `src/linear_algebra/` | Member B | Member A | Covariance, expected returns |
| `src/optimization/` | Member B | Member A | Mean-variance, Max Sharpe, Risk Parity |
| `src/risk/` | Member A | Member B | VaR, CVaR, Sharpe, drawdown |
| `src/visualization/` | Member B | Member A | Plotly charts |
| `app/` (Streamlit) | Both | Both | Integration, UI |
| `tests/` | Both | Both | pytest for each module |
| `docs/` | Both | Both | Screenshots, final README |

## Collaboration Rules

1. **Main branch protected** – PRs for Phase 1+ features
2. **Environment parity** – Both use Codespaces with same devcontainer.json, Python 3.14.2, pinned requirements
3. **Git hygiene** – .venv/, __pycache__/, *.csv ignored; .gitkeep tracks empty data folders
4. **Verification before merge** – pip list matches pins, smoke test passes, git status clean

## Phase 0 Completion Evidence

- Skeleton matches reference tree, all __init__.py present
- requirements.txt pinned: numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, yfinance 1.4.1, plotly 7.1.0, streamlit 1.66.1, pytest 9.2.0
- .venv/ created at repo root, interpreter path: /workspaces/.../.venv/bin/python
- devcontainer rebuild succeeds, auto-installs requirements, forwards 8501
- .gitignore verified: !! .venv/, !! __pycache__/, !! *.csv
- README and LICENSE render on GitHub, setup verified end-to-end
- Smoke test: all 7 imports OK
- Initial commits pushed: a606e60 HEAD -> main

## Next Phase

Phase 1: Data Ingestion – yfinance download to data/raw/



---

## Data Universe (Phase 1.1)

### Constants Location
`src/data/stock_data.py` - single source of truth for all phases

```python
DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "XOM", "AMZN"]
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None  # resolves to today at call time
TRADING_DAYS_PER_YEAR = 252
```

### Ticker Universe - 7 Tickers Across 6 Sectors

| Ticker | Company | Sector (GICS) | Market Cap | Rationale |
|--------|---------|---------------|------------|-----------|
| AAPL | Apple Inc. | Technology (Consumer Electronics) | Large-cap >$2T | Most liquid US stock, 10y+ clean history, tech anchor |
| MSFT | Microsoft Corp. | Technology (Software - Infrastructure) | Large-cap >$2T | Diversified tech (cloud, software), different sub-sector from AAPL |
| JPM | JPMorgan Chase & Co. | Financial Services | Large-cap >$500B | Largest US bank, represents financials, interest-rate sensitive |
| JNJ | Johnson & Johnson | Healthcare (Pharma) | Large-cap >$350B | Defensive healthcare, low beta, dividend aristocrat |
| PG | Procter & Gamble | Consumer Staples | Large-cap >$350B | Defensive consumer staples, low correlation with tech |
| XOM | Exxon Mobil Corp. | Energy | Large-cap >$400B | Largest US energy, provides inflation/commodity hedge, low correlation with tech |
| AMZN | Amazon.com Inc. | Consumer Discretionary (Internet Retail) | Large-cap >$1.5T | E-commerce/cloud, high growth, consumer discretionary exposure |

**Sector Diversity:** 6 distinct sectors (Technology x2 different industries, Financials, Healthcare, Consumer Staples, Energy, Consumer Discretionary) – exceeds requirement of ≥3 sectors.

**Validation:**
- All 7 have Adj Close data back to 2008+ (well before 2018-01-01 start)
- No penny stocks, no ETFs with structure changes, no recent IPOs
- Average daily volume >10M shares (high liquidity)
- Manual Yahoo Finance check: https://finance.yahoo.com/quote/<TICKER>/history/

### Date Range

- `DEFAULT_START_DATE = "2018-01-01"` – 8+ years of data, includes COVID crash (2020), inflation cycle (2022), rate hikes, provides regime diversity
- `DEFAULT_END_DATE = None` – at runtime resolves to `datetime.today()` or `yfinance` default (today), ensuring always up-to-date but deterministic in code
- `TRADING_DAYS_PER_YEAR = 252` – industry standard annualization constant

### Annualization Convention (for Phases 2-5)

```
mu_annual = mu_daily × 252
Sigma_annual = Sigma_daily × 252
sigma_annual = sigma_daily × √252
Sharpe_annual = Sharpe_daily × √252
```

Fixing 252 here avoids classic bugs of using 250, 252, 260, or 365 inconsistently across modules.

## Return Convention (Phase 2.1)

**Convention chosen:** Simple returns (`RETURN_TYPE = "simple"`), defined in `src/preprocessing/returns.py`.

**Formulas:**
- Simple return: r_t = (P_t - P_{t-1}) / P_{t-1} = P_t / P_{t-1} - 1
- Log return: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1}) [documented but not used]

**Why simple:**
1. Markowitz linearity: R_p = w^T mu exact only for simple returns.
2. Covariance identity Sigma = X^T X/(T-1) where X is centered simple-return matrix.
3. Quadratic forms w^T Sigma w require linear aggregation.
4. Log variant would be separate helper `compute_log_returns`, not wired to pipeline.

**Annualization (single source of truth):**
- TRADING_DAYS_PER_YEAR = 252 imported from src.data.stock_data
- mu_annual = mu_daily * 252
- Sigma_annual = Sigma_daily * 252
- sigma_annual = sigma_daily * sqrt(252)

**Verification:** from src.preprocessing.returns import RETURN_TYPE, TRADING_DAYS_PER_YEAR; assert RETURN_TYPE in {"simple","log"} and TRADING_DAYS_PER_YEAR == 252

### Volatility Convention (Phase 2.4)

**Function:** `compute_volatility(returns, annualize=True) -> pd.Series` in `src/preprocessing/returns.py`.

**Formula:**
- sigma_i = sqrt( (1/(T-1)) sum_t (r_{i,t} - mu_i)^2 ) – sample std, ddof=1.
- Annualized: sigma_annual = sigma_daily * sqrt(252)

**Why sqrt(252) not 252:**
Variance adds linearly under independence: Var(annual) ≈ 252 * Var(daily). Std = sqrt(Var), so sqrt(252*Var) = sqrt(252)*sigma. Covariance Sigma annualizes by *252, volatility by *sqrt(252). Mixing these is a common undergrad error.

**DDOF Contract (critical for Phase 3):**
- Phase 2 volatility uses ddof=1 (pandas .std() default).
- Phase 3 covariance Sigma = X^T X / (T-1) uses T-1 denominator.
- If Phase 2 used ddof=0, sqrt(diag(Sigma_annual)) would disagree with sigma_annual by sqrt((T-1)/T).
- Contract verified: np.isclose(sigma_annual, sqrt(var(ddof=1)*252)) and later sqrt(diag(Sigma_annual)) == sigma_annual elementwise.

**Verification:** returns.std(ddof=1)*sqrt(252) exact match.

---

## Phase 7 Status Update (2026-11-14)

All phases (0–6) are complete. The `src/` ownership plan documented above was
honored throughout: the `src/data/` and `src/preprocessing/` owners led
Phases 1–2; the `src/linear_algebra/` and `src/optimization/` owners led
Phases 3–4; both members contributed to Phases 5–6 as the risk and
dashboard layers crossed module boundaries.

This file is preserved as a historical record of the original plan. The
authoritative project structure is documented in `README.md`. The final test
baseline is 252 passing tests documented in `docs/phase7_test_report.md`.