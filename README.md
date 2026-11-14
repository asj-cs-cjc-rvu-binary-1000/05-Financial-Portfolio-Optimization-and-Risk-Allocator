# Financial Portfolio Optimization and Risk Allocator

A quantitative finance application that downloads historical market data, estimates risk-return characteristics, and constructs optimal portfolios using Modern Portfolio Theory. The app allows users to select tickers, estimate expected returns and covariance, run optimization (Mean-Variance, Max Sharpe, Target Return), analyze risk metrics (VaR, CVaR, drawdown), and visualize results through an interactive Streamlit dashboard.

**Target audience:** Students and practitioners learning portfolio optimization, risk management, and applied linear algebra / optimization in Python. Built as an 8-phase academic project (Phases 0–7).

## Team Members

| Name | Email | USN |
| :--- | :--- | :--- |
| Bibek Panigrahi | bibekpanigrahibtech25@rvu.edu.in | 1RUA25CSE0121 |
| Aryan S Jain | aryansjainbtech25@rvu.edu.in | 1RUA25CSE0087 |



## Tech Stack

Pinned versions verified on Python 3.14.10 (Nov 2026). See `docs/environment_versions.md` for the pin history and rationale.

- **numpy==2.5.4** – Linear algebra, vectorized calculations
- **pandas==3.1.0** – Time series handling, data manipulation
- **scipy==1.18.1** – Optimization solvers (`minimize` with SLSQP), statistical functions
- **yfinance==1.4.1** – Market data download from Yahoo Finance
- **plotly==6.3.0** – Interactive visualizations (approved viz library). Paired with **kaleido==0.2.1** for headless PNG export without system Chrome.
- **streamlit==1.66.1** – Web app framework for portfolio allocator
- **pytest==9.2.0** – Unit and integration testing
- **nbformat==5.10.4**, **ipykernel==6.29.5** – Notebook rendering support
- **tabulate==0.9.0** – Markdown table formatting for CLI reports

> No databases, cloud SDKs, or unapproved frameworks. Matplotlib omitted intentionally – Plotly is the approved visualization library.

## Repository Structure

Reference structure from Section 3 of project brief:

```
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   ├── raw/              # Downloaded CSVs (ignored, kept via .gitkeep)
│   │   └── .gitkeep
│   └── processed/        # Cleaned / processed datasets (ignored)
│       └── .gitkeep
├── notebooks/            # Exploratory notebooks
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── stock_data.py         # Phase 1: download, clean, save/load
│   ├── preprocessing/
│   │   └── returns.py            # Phase 2: daily returns, μ, σ
│   ├── linear_algebra/
│   │   ├── matrix_ops.py         # Phase 3: center, transpose, quad form
│   │   ├── covariance.py         # Phase 3: Σ, Corr, PSD checks
│   │   └── eigen_analysis.py     # Phase 3: eigendecomposition
│   ├── optimization/
│   │   ├── portfolio_metrics.py  # Phase 4: R_p, σ², σ, Sharpe
│   │   ├── constraints.py        # Phase 4: budget, long-only, target
│   │   ├── benchmarks.py         # Phase 4: equal-weight
│   │   ├── optimizer.py          # Phase 4: min-var, max-Sharpe, target-return
│   │   └── optimization_pipeline.py  # Phase 4: compare_portfolios + CLI
│   ├── risk/
│   │   ├── __init__.py           # Phase 5: package exports
│   │   ├── metrics.py            # Phase 5: drawdown, VaR, CVaR
│   │   ├── frontier.py           # Phase 5: efficient frontier
│   │   └── comparison.py         # Phase 5: compare_risk_metrics + CLI        # Phase 5: VaR, CVaR, drawdown
│   └── visualization/
│       └── charts.py             # Phase 3+: Plotly charts
├── app/                          # Streamlit app (Phase 6)
│   ├── __init__.py
│   └── app.py                    # entry point — `streamlit run app/app.py`
├── tests/                        # pytest suite (254 tests as of Phase 6)
│   └── __init__.py
├── docs/
│   ├── mathematical_foundations.md   # Phases 3 & 4 LaTeX derivations
│   ├── optimization_specification.md # Phase 4 decisions Q1–Q7
│   ├── optimization_summary.md       # Phase 4 live comparison table
│   ├── eigen_analysis.md             # Phase 3 live eigen report
│   ├── return_summary.md             # Phase 2 live return report
│   ├── data_summary.md               # Phase 1 live data report
│   └── screenshots/                  # PNGs for final README
├── pytest.ini                    # Pin sys.path=repo root for tests
├── requirements.txt
└── README.md
```

Every Python package folder contains an empty `__init__.py` so imports like `from src.linear_algebra import covariance` work.

## Setup Instructions

### Option A: GitHub Codespaces (Recommended)

1. Clone the repo:
   ```bash
   git clone https://github.com/aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator.git
   cd 05-Financial-Portfolio-Optimization-and-Risk-Allocator
   ```
2. Open in Codespace: `Code` → `Create codespace on main`
3. Wait for container build – `postCreateCommand` automatically runs:
   ```bash
   python -m pip install --upgrade pip && pip install -r requirements.txt
   ```
4. Verify:
   ```bash
   python --version
   # Python 3.14.x
   pip list | grep -E "numpy|pandas|scipy|yfinance|plotly|streamlit|pytest"
   ```

No manual pip install needed – environment is auto-provisioned.

### Option B: Local Setup

Tested on Python 3.14.10:

```bash
git clone https://github.com/aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator.git
cd 05-Financial-Portfolio-Optimization-and-Risk-Allocator

# 1. Create venv at repo root
python3 -m venv .venv

# 2. Activate
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install pinned deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
which python
# Should point to .venv/bin/python
python -c "import sys; print(sys.executable)"
pip list

# 5. Test persistence
deactivate
source .venv/bin/activate
python --version
```

`.venv/` is ignored by `.gitignore`. Generated CSVs in `data/raw/*.csv` and `data/processed/*.csv` are also ignored – folders stay tracked via `.gitkeep`.

## How to Run

### Phase 1 – Data Ingestion (Complete)

**One-command data refresh (for both team members):**

```bash
source .venv/bin/activate
python -m src.data.stock_data
```

This command:
- Downloads 7 tickers (AAPL, MSFT, JPM, JNJ, PG, XOM, AMZN) from 2018-01-01 to today via yfinance
- Cleans (drops all-NaN tickers, ffill limit 5, dropna to preserve PSD)
- Saves raw to `data/raw/prices_raw_<tickers>_<start>_<end>.csv` and cleaned to `data/processed/prices_clean_<tickers>_<start>_<end>.csv` (ignored by Git, folders tracked via .gitkeep)
- Validates (checks positive, no NaN, duplicate dates, trading days/year ~252)
- Prints summary and saves Markdown report to `docs/data_summary.md`

**Other commands:**

```bash
pytest tests/test_stock_data.py -v   # 10 passed
```

### Phase 2 – Returns (Complete)

```bash
python -m src.preprocessing.returns
# Loads latest data/processed/prices_clean_*.csv, computes returns, μ, σ,
# writes docs/return_summary.md
```

```bash
pytest tests/test_returns.py -v   # 10 passed
```

### Phase 3 – Linear Algebra: Covariance, Correlation, Eigen (Complete)

**Why Linear Algebra matters for finance:** Portfolio variance is exactly the quadratic form $w^\top \Sigma w$. Minimizing it is the Markowitz problem. This phase proves all identities from-scratch.

**Notation:** $X \in \mathbb{R}^{T\times N}$ returns, $\tilde{X}=X-\mathbf{1}_T\mu^\top$ centered, $\Sigma = \tilde{X}^\top\tilde{X}/(T-1)$, $D=\text{diag}(\sigma)$, $\text{Corr}=D^{-1}\Sigma D^{-1}$, $\Sigma=V\Lambda V^\top$

**Key functions** (in `src/linear_algebra/`):
- `matrix_ops.py` — `center_columns`, `transpose`, `matrix_multiply`, `quadratic_form(w, Σ) = wᵀΣw`, `is_symmetric`
- `covariance.py` — `compute_covariance_matrix` (from $\tilde{X}^\top\tilde{X}/(T-1)$, no `DataFrame.cov()`), `compute_correlation_matrix` ($D^{-1}\Sigma D^{-1}$), `verify_covariance_properties`
- `eigen_analysis.py` — `eigen_decompose` (`eigh`, descending), `eigen_summary_table`, `first_principal_direction`

**Visualizations** (in `src/visualization/charts.py`, Plotly-only):
- `plot_correlation_heatmap(Corr)` — square RdBu centered 0 range [-1,1], 2-decimal annotations
- `plot_eigenvalue_spectrum(eigen_dict)` — bar eigenvalues descending + cumulative explained on secondary y-axis

| Correlation Heatmap | Eigenvalue Spectrum |
| :---: | :---: |
| [Correlation](docs/screenshots/corr_heatmap.png) | [Eigen Spectrum](docs/screenshots/eigen_spectrum.png) |

**How to run Phase 3:**

```bash
python - << 'PY'
from src.data.stock_data import load_prices
from src.preprocessing.returns import process_returns_pipeline
from src.linear_algebra.covariance import compute_covariance_matrix, compute_correlation_matrix, verify_covariance_properties
from src.linear_algebra.eigen_analysis import eigen_decompose
from pathlib import Path
latest = max(Path("data/processed").glob("prices_clean_*.csv"), key=lambda p: p.stat().st_mtime)
rets, mu, sigma, _ = process_returns_pipeline(load_prices(str(latest)))
Sigma_daily = compute_covariance_matrix(rets, annualize=False)
Sigma_annual = compute_covariance_matrix(rets, annualize=True)
print(verify_covariance_properties(Sigma_daily, Sigma_annual, sigma))
PY

# Regenerate PNGs (uses kaleido==0.2.1, no system Chrome needed)
python -m src.visualization.charts

# Offline tests
pytest tests/test_covariance.py -v   # 17 passed
```

### Phase 4 – Optimization (Complete)

**Why Linear Algebra matters here:** the Markowitz problem is a *quadratic program* — minimize $w^\top \Sigma w$ subject to $1^\top w = 1$ and $0 \le w_i \le 1$. It uses every Phase 3 primitive (quadratic form, covariance matrix) and the SciPy SLSQP solver. Full derivation in `docs/mathematical_foundations.md` §11.

**Key functions** (in `src/optimization/`):
- `portfolio_metrics.py` — `portfolio_return(w, μ)` = $w^\top\mu$, `portfolio_variance(w, Σ)` = $w^\top\Sigma w$ (routes through Phase 3 `quadratic_form`), `portfolio_volatility`, `portfolio_sharpe`
- `constraints.py` — `validate_weights` (post-optimization gate), `budget_constraint`, `long_only_bounds`, `target_return_constraint`
- `benchmarks.py` — `equal_weights`, `equal_weight_portfolio`
- `optimizer.py` — `minimum_variance_portfolio`, `maximum_sharpe_portfolio`, `target_return_portfolio` (all use `scipy.optimize.minimize`, `method="SLSQP"`)
- `optimization_pipeline.py` — `compare_portfolios` (single comparison table)

**One-command comparison (on the latest processed CSV):**

```bash
source .venv/bin/activate
python -m src.optimization.optimization_pipeline
```

**Example output on real 7-ticker data (r_f = 2%/yr):**

```
               return  volatility  variance  sharpe  success  n_iter    AAPL    MSFT     JPM     JNJ      PG     XOM    AMZN
EqualWeight    0.1988      0.1843    0.0340  0.9701     True       0  0.1429  0.1429  0.1429  0.1429  0.1429  0.1429  0.1429
MinVolatility  0.1479      0.1587    0.0252  0.8054     True      25  0.0000  0.0395  0.0284  0.3520  0.3383  0.1364  0.1055
MaxSharpe      0.2236      0.2018    0.0407  1.0088     True      10  0.3378  0.2242  0.0825  0.2075  0.0296  0.1055  0.0130
```

**Reading the table:**
- **EqualWeight** gives every ticker 1/7 — the naive benchmark
- **MinVolatility** concentrates on low-vol defensive names (JNJ + PG = 69%), achieves the lowest volatility (15.9% vs 18.4% equal-weight)
- **MaxSharpe** concentrates on high-Sharpe tech (AAPL + MSFT = 56%), achieves the highest Sharpe (1.009 vs 0.970 equal-weight)

This is the canonical Markowitz result: **the two optimizers produce genuinely different portfolios**, each optimal for its own objective. Full interpretation is auto-written to `docs/optimization_summary.md` on each run.

**Offline tests:**

```bash
pytest tests/test_optimizer.py -v            # 45 passed
pytest tests/test_optimization_pipeline.py -v # 13 passed
pytest tests/ -v                             # 160 passed (all phases)
```

**Note on `RuntimeWarning`:** running `python -m src.optimization.optimization_pipeline` may print a `RuntimeWarning` about a module appearing in `sys.modules` before execution. This is expected Python behavior when a package `__init__.py` imports a module that is also run via `-m`. It is harmless and the CLI runs correctly.

### Phase 5 – Risk Analysis & Efficient Frontier (Complete)

**Why this matters:** Phase 4 gave us the *optimizers*. Phase 5 gives us the *diagnostics* — how bad can a bad day be (VaR), how bad is a bad day on average (CVaR), how deep was the worst peak-to-trough drawdown (MDD), and what does the whole risk-return landscape look like (the efficient frontier). Full derivations in `docs/mathematical_foundations.md` §12.

**Key functions** (in `src/risk/`):
- `metrics.py` — `drawdown_series`, `maximum_drawdown` (signed ≤ 0), `historical_var` (positive loss magnitude), `historical_cvar`, `portfolio_historical_returns` (fixed weights, daily rebalancing)
- `frontier.py` — `efficient_frontier(mu, Sigma, n_points=50)` returns `(efficient_df, inefficient_df, metadata)`; `analyze_frontier(...)` bundles the two branches with the min-var / max-Sharpe / equal-weight anchor portfolios and per-asset scatter data
- `comparison.py` — `compare_risk_metrics(mu, Sigma, returns, r_f, ...)` adds `var_95`, `cvar_95`, `max_drawdown` to the Phase 4 comparison table

**Visualizations** (in `src/visualization/charts.py`, Plotly-only):
- `plot_efficient_frontier(analysis)` — six-layer figure: inefficient dashed grey, efficient solid blue, per-asset scatter, labeled Min Volatility / Max Sharpe / Equal Weight points
- `plot_drawdown(drawdown_series, portfolio_name)` — filled area chart, MDD annotated
- `plot_var_histogram(returns, var_value, portfolio_name, confidence_level=0.95)` — two-trace histogram (body / tail) with the VaR threshold line

**One-command risk comparison (on the latest processed CSV):**

```bash
source .venv/bin/activate
python -m src.risk.comparison
```

Example output on real 7-ticker data (r_f = 2%/yr, c = 95%):

```text
               return  volatility  sharpe  var_95  cvar_95  max_drawdown  success    AAPL    MSFT     JPM     JNJ      PG     XOM    AMZN
EqualWeight    0.1988      0.1843  0.9701  0.0166   0.0277       -0.3019     True  0.1429  0.1429  0.1429  0.1429  0.1429  0.1429  0.1429
MinVolatility  0.1479      0.1587  0.8054  0.0139   0.0231       -0.2748     True  0.0000  0.0395  0.0284  0.3520  0.3383  0.1364  0.1055
MaxSharpe      0.2236      0.2018  1.0088  0.0184   0.0299       -0.3142     True  0.3378  0.2242  0.0825  0.2075  0.0296  0.1055  0.0130
```

Reading the table:

- VaR / CVaR are positive loss magnitudes (industry convention). 1.66% VaR for EqualWeight means "on 95% of days the portfolio's loss does not exceed 1.66%."

- MDD is a signed peak-to-trough cumulative return, always ≤ 0. -30.19% for EqualWeight means "the portfolio's worst peak-to-trough decline over 2018–2026 was 30.19%."

- MinVolatility has the lowest VaR (1.39%), lowest CVaR (2.31%), and shallowest drawdown (-27.48%) — exactly what a defensive portfolio should produce. It also has the lowest Sharpe (0.81) because it sacrifices return for safety.

- MaxSharpe has the highest Sharpe (1.0088) but the worst tail risk (VaR 1.84%, CVaR 2.99%, MDD -31.42%) — the classic return-for-risk trade-off in the tail.

- Full interpretation is auto-written to docs/risk_analysis.md on each run.

Regenerate all five PNGs (Phase 3 + Phase 5):

```bash
python -m src.visualization.charts
Offline tests:
```

```bash
pytest tests/test_risk_metrics.py -v          # 23 passed
pytest tests/test_frontier.py -v              # 20 passed
pytest tests/test_risk_comparison.py -v       # 12 passed
pytest tests/test_visualization.py -v         # 19 passed
pytest tests/ -v                              # 234 passed (all phases)
```

Note on kaleido DeprecationWarning: python -m src.visualization.charts prints a DeprecationWarning about kaleido < 1.0.0. This is expected: kaleido==0.2.1 is pinned because kaleido >= 1.0 requires system Chrome, which fails in the Codespace sandbox. The pinned version works headless and renders all five PNGs correctly. Do not upgrade kaleido — see docs/environment_versions.md for the Phase 3 rationale.

### Phase 5 visuals

| Efficient Frontier | Drawdown (Max Sharpe) | VaR Histogram (Max Sharpe) |
| :---: | :---: | :---: |
| [Efficient Frontier](docs/screenshots/efficient_frontier.png) | [Drawdown](docs/screenshots/drawdown_max_sharpe.png) | [VaR Histogram](docs/screenshots/var_histogram_max_sharpe.png) |

### Phase 6 dashboard

| Optimization tab | Risk Analysis tab | Frontier tab |
| :---: | :---: | :---: |
| [Optimization](docs/screenshots/dashboard_optimization.png) | [Risk Analysis](docs/screenshots/dashboard_risk.png) | [Frontier](docs/screenshots/dashboard_frontier.png) |

### Phase 6 – Interactive Streamlit Dashboard (Complete)

**Why this matters:** Phases 1–5 built a full analytical pipeline. Phase 6 exposes it through a browser UI so a grader or user can change inputs and see the numbers update — without ever touching Python. The dashboard is a **read-only rendering layer** over the Phase 4 and Phase 5 public functions. No mathematical logic is reimplemented inside Streamlit.

**Dashboard structure** (`app/app.py`, single file, ~600 lines):
- **Sidebar** — ticker multiselect from the CSV, date range, risk-free rate slider, VaR/CVaR confidence, frontier resolution, optional target-return slider (bounds dynamically from selected assets).
- **Tab 1 — Optimization:** `compare_portfolios` → comparison table + grouped allocation bar chart.
- **Tab 2 — Risk Analysis:** `compare_risk_metrics` → risk table + portfolio selector + `plot_drawdown` + `plot_var_histogram` + VaR/CVaR/MDD metric cards.
- **Tab 3 — Frontier:** `analyze_frontier` → six-layer efficient frontier chart (`plot_efficient_frontier`) + anchor cards + metadata table.

**Caching:** four `@st.cache_data` boundaries on `_load_prices`, `_returns_full`, `_filtered_data`, `_analyze_frontier`. The last is the most expensive (~2–4s per call, 100 SLSQP optimizations) and is critical to a smooth UX.

**Error handling:** three-layer model — pre-run validation (sidebar errors + disabled Run), per-tab runtime catch (no `st.stop()`; a failure in one tab does not break the others), post-run success-flag inspection (the optimizer's own `message` field is surfaced to the user).

**How to run:**

```bash
source .venv/bin/activate
streamlit run app/app.py
# Serves on http://localhost:8501 (forwarded by devcontainer from Phase 0)
```

Open the forwarded URL in the Codespace Ports tab.

**Prerequisites:** data/processed/prices_clean_*.csv must exist. If missing, run python -m src.data.stock_data first. The dashboard does not download data itself.

**Tests:** pytest tests/test_dashboard.py -v → 20 passed (pure helpers + validation + target bounds). The Streamlit widgets themselves are visually verified — no browser automation.

**Screenshots:** docs/screenshots/dashboard_optimization.png, dashboard_risk.png, dashboard_frontier.png — captured from the running app on the real 7-ticker dataset at r_f = 2%.

### Phase 7 — Pending
Documentation finalization, final screenshots, final testing, presentation

## Team Members and Course Context

- **Repository:** `aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator`
- **Course/Project:** Financial Portfolio Optimization – 8-phase implementation (Phases 0–7)
  - Phase 0: Environment & Skeleton — ✅ Complete
  - Phase 1: Data Ingestion (yfinance) — ✅ Complete
  - Phase 2: Preprocessing & Returns — ✅ Complete
  - Phase 3: Linear Algebra (Covariance, Correlation, Eigen) — ✅ Complete
  - Phase 4: Optimization (Mean-Variance, Max-Sharpe, Target-Return) — ✅ Complete
  - Phase 5: Risk Analysis (VaR, CVaR, Drawdown) & Efficient Frontier — ✅ Complete
  - Phase 6: Interactive Streamlit Dashboard — ✅ Complete
  - Phase 7: Documentation, Testing & Final Polish — ⏳ Pending

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Financial Portfolio Optimization and Risk Allocator Team. Academic coursework use.