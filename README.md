# 05 - Financial Portfolio Optimization and Risk Allocator

A quantitative finance application that downloads historical market data, estimates risk-return characteristics, and constructs optimal portfolios using Modern Portfolio Theory. The app allows users to select tickers, estimate expected returns and covariance, run optimization (Mean-Variance, Risk Parity, Max Sharpe), analyze risk metrics (VaR, CVaR, drawdown), and visualize results through an interactive Streamlit dashboard.

**Target audience:** Students and practitioners learning portfolio optimization, risk management, and applied linear algebra / optimization in Python. Built as an 8-phase academic project.

## Team Members

| Name | Email | USN |
| :--- | :--- | :--- |
| Bibek Panigrahi | bibekpanigrahibtech25@rvu.edu.in | 1RUA25CSE0121 |
| Aryan S Jain | aryansjainbtech25@rvu.edu.in | 1RUA25CSE0087 |



## Tech Stack

Pinned versions verified on Python 3.14.2 (Oct 2026):

- **numpy==2.5.4** – Linear algebra, vectorized calculations
- **pandas==3.1.0** – Time series handling, data manipulation
- **scipy==1.18.1** – Optimization solvers (minimize), statistical functions
- **yfinance==1.4.1** – Market data download from Yahoo Finance
- **plotly==7.1.0** – Interactive visualizations (approved viz library)
- **streamlit==1.66.1** – Web app framework for portfolio allocator
- **pytest==9.2.0** – Unit and integration testing

> No databases, cloud SDKs, or unapproved frameworks. Matplotlib omitted intentionally – Plotly is the approved visualization library.

## Repository Structure

Reference structure from Section 3 of project brief:

```
.
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
│   │   └── __init__.py   # Data download & loading
│   ├── preprocessing/
│   │   └── __init__.py   # Returns, cleaning
│   ├── linear_algebra/
│   │   └── __init__.py   # Covariance, expected returns
│   ├── optimization/
│   │   └── __init__.py   # Mean-variance, risk parity, etc.
│   ├── risk/
│   │   └── __init__.py   # VaR, CVaR, drawdown, Sharpe
│   └── visualization/
│       └── __init__.py   # Plotly charts
├── app/                  # Streamlit app (Phase 6)
├── tests/                # pytest suite
│   └── __init__.py
├── docs/
│   └── screenshots/      # Screenshots for final README
├── .gitignore
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
   # Python 3.14.2
   pip list | grep -E "numpy|pandas|scipy|yfinance|plotly|streamlit|pytest"
   ```

No manual pip install needed – environment is auto-provisioned.

### Option B: Local Setup

Tested on Python 3.14.2:

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

**Example output:**
```
Raw shape: (2220, 7)
Clean shape: (2220, 7), Dropped: []
Validation Summary:
  n_tickers: 7, n_rows: 2220, trading_days/year avg: 251.6
  all_positive: True, has_nan: False, duplicate_dates: False
```

**Other commands:**

```bash
# Run offline unit tests (no Yahoo call)
pytest tests/test_stock_data.py -v
# Expected: 10 passed

# Use caching in code
from src.data.stock_data import fetch_and_cache
df = fetch_and_cache(["AAPL","MSFT"], "2020-01-01", None)  # loads from cache if exists

# Exploratory notebook
# Open notebooks/exploratory_analysis.ipynb in VS Code Jupyter
# Run All – Plotly charts render with HTML(fig.to_html()) workaround

# Round-trip test
python - << 'PY'
from src.data.stock_data import save_prices, load_prices
import pandas as pd
df = pd.DataFrame({"AAPL":[100.1]}, index=pd.date_range("2023-01-01", periods=1))
save_prices(df, "data/raw/test.csv")
print(load_prices("data/raw/test.csv").equals(df))
PY
```

**Phase 6 placeholder:** Streamlit app will be added in `app/`:

```bash
# Future command (Phase 6):
streamlit run app/app.py
# Served on port 8501 (forwarded in devcontainer)
```


### Phase 2 – Returns (Complete)

```bash
python -m src.preprocessing.returns
# Loads latest data/processed/prices_clean_*.csv, computes returns (2219,7), mu, sigma, writes docs/return_summary.md
# Sanity: AAPL sigma 0.20-0.40, JNJ 0.12-0.22
```

```bash
pytest tests/test_returns.py -v
# 10 passed
```

### Phase 3 – Linear Algebra: Covariance, Correlation, Eigen (Complete)

**Why Linear Algebra matters for finance:** Portfolio variance is exactly the quadratic form $w^\top \Sigma w$. Minimizing it is the Markowitz problem. This phase proves all identities from-scratch.

**Notation:** $X \in \mathbb{R}^{T\times N}$ returns, $\tilde{X}=X-\mathbf{1}_T\mu^\top$ centered, $\Sigma = \tilde{X}^\top\tilde{X}/(T-1)$, $D=\text{diag}(\sigma)$, $\text{Corr}=D^{-1}\Sigma D^{-1}$, $\Sigma=V\Lambda V^\top$

#### Tasks

**3.1 Matrix utilities** `src/linear_algebra/matrix_ops.py`
- `center_columns`, `transpose`, `matrix_multiply`, `quadratic_form(w,Σ)=wᵀΣw`, `is_symmetric`
- Shape assertions: $(N\times T)(T\times N)=N\times N$

**3.2 Covariance** `src/linear_algebra/covariance.py`
- Derivation: $\text{Cov}(X_i,X_j)=E[(X_i-\mu_i)(X_j-\mu_j)]$ → sample $(1/(T-1))Σ_t(r_{t,i}-μ_i)(r_{t,j}-μ_j)$ → matrix $Σ_{\text{daily}}=\tilde{X}^\top\tilde{X}/(T-1)$
- Annualized $Σ_{\text{annual}}=252·Σ_{\text{daily}}$
- No `np.cov`, no `DataFrame.cov()` in impl — uses Task 3.1 primitives
- Matches `returns.cov(ddof=1)` within 1e-12 (oracle test)

**3.3 Symmetry & PSD**
- Symmetry: $Σ_{ij}=Σ_{ji}$ by commutativity → $Σ=Σᵀ$, $||Σ-Σᵀ||_{max}≤1e-10$
- PSD: $wᵀΣw = 1/(T-1)||\tilde{X}w||² ≥0$, Gram matrix → $λ_i≥0$, `eigvalsh` check min_eig≥-1e-12
- `sigma_from_covariance`: $σ=√{\text{diag}(Σ_{\text{annual}})}$, matches Phase 2 `compute_volatility` within 1e-10 (hard requirement)
- Condition number $κ=λ_{max}/λ_{min}$: 10–200 moderate for 7 tickers

**3.4 Correlation**
- $\text{Corr}_{ij}=Σ_{ij}/(σ_iσ_j)$, matrix $\text{Corr}=D^{-1}ΣD^{-1}$, $D=\text{diag}(σ)$
- diag=1, ∈[-1,1], symmetric PSD, unit-free
- Matches `returns.corr()` within 1e-12, $252$ cancels so annual=daily

**3.5 Eigendecomposition**
- Spectral theorem: $Σ=VΛVᵀ=Σ_k λ_k v_k v_kᵀ$, $V$ orthogonal $VᵀV=I$, $λ_1≥…≥λ_N≥0$
- Uses `np.linalg.eigh` (Hermitian), descending, sign convention largest-magnitude positive
- Variance explained: $\text{Tr}(Σ)=Σ_kλ_k$, $\text{explained}_k=λ_k/\text{Tr}(Σ)$, cumulative
- Reconstruction $VΛVᵀ≈Σ$ within 1e-10, orthonormal $VᵀV≈I$ within 1e-10
- First PC often all same sign — market mode

**3.6 Summary table & docs**
- `eigen_summary_table(Σ)` → (7,10) DataFrame: PC, eigenvalue, explained_ratio, cumulative_ratio, eigenvector weights
- `first_principal_direction` → $v_1$
- `docs/eigen_analysis.md` live run: trace, eigenvalues, explained 48% +19% +9%..., condition, first 3 PCs 77% → 3 independent factors
- `docs/mathematical_foundations.md` — full LaTeX derivations (see Section 9)

**3.7 Visualization** `src/visualization/charts.py`
- `plot_correlation_heatmap(Corr)`: square, RdBu centered 0 range [-1,1], ticker labels, annotations 2 decimals diagonal 1.00
- `plot_eigenvalue_spectrum(eigen_dict)`: bar eigenvalues descending + cumulative explained secondary y-axis ending 1.0
- Plotly-only (approved), no seaborn/matplotlib
- PNGs `docs/screenshots/corr_heatmap.png` (148KB, 600×600) + `eigen_spectrum.png` (130KB, 700×500) via `plotly==6.3.0` + `kaleido==0.2.1` (no Chrome/Brave needed)

| Correlation Heatmap | Eigenvalue Spectrum |
| :---: | :---: |
| [Correlation](docs/screenshots/corr_heatmap.png) | [Eigen Spectrum](docs/screenshots/eigen_spectrum.png) |

**3.8 Offline unit tests** `tests/test_covariance.py`
- 17 tests, all offline, no yfinance (`conftest.py` mocks)
- Oracle checks: Σ matches `returns.cov(ddof=1)`, Corr matches `returns.corr()`, eigenvalues match `eigvalsh(Σ)[::-1]`
- Internal: $√{\text{diag}(Σ_{\text{annual}})}==σ$, $VΛVᵀ≈Σ$, $VᵀV≈I$, explained sum 1, $wᵀΣw==wΣw$, rejects non-symmetric, center_columns zero means
- Break ddof=1→0 → tests 1,5 fail as expected, revert → green

**3.9 Mathematical foundations** `docs/mathematical_foundations.md`
- Sections: Notation, Covariance derivation, Symmetry, PSD, Eigendecomposition, Correlation as diagonal scaling, Annualization table, Why finance ($wᵀΣw$ → Markowitz), Implementation map

#### How to Run Phase 3

```bash
source .venv/bin/activate

# Install pinned stack (plotly 6.3 + kaleido 0.2.1 needed for PNGs without Chrome)
pip install -r requirements.txt
# numpy==2.5.4 pandas==3.1.0 scipy==1.18.1 yfinance==1.4.1 plotly==6.3.0 kaleido==0.2.1 tabulate==0.9.0 streamlit==1.66.1 pytest==9.2.0

# 1. Compute covariance/correlation/eigen from latest processed CSV
python - << 'PY'
from src.data.stock_data import load_prices
from src.preprocessing.returns import process_returns_pipeline
from src.linear_algebra.covariance import compute_covariance_matrix, compute_correlation_matrix, verify_covariance_properties
from src.linear_algebra.eigen_analysis import eigen_decompose, eigen_summary_table
from pathlib import Path
latest = max(Path("data/processed").glob("prices_clean_*.csv"), key=lambda p: p.stat().st_mtime)
prices = load_prices(latest)
rets, mu, sigma, _ = process_returns_pipeline(prices)
Sigma = compute_covariance_matrix(rets, annualize=False)
Sigma_a = compute_covariance_matrix(rets, annualize=True)
Corr = compute_correlation_matrix(Sigma)
eig = eigen_decompose(Sigma)
print(f"Sigma shape {Sigma.shape}, symmetric {Sigma.index.equals(Sigma.columns)}, min_eig {eig['values'][-1]:.2e}")
print(verify_covariance_properties(Sigma, Sigma_a, sigma))
print(eigen_summary_table(Sigma_a).head())
PY

# 2. Generate PNGs (no Chrome needed)
python -m src.visualization.charts
# Saved docs/screenshots/corr_heatmap.png via kaleido 0.2.1 (148KB)
# Saved docs/screenshots/eigen_spectrum.png (130KB)

# 3. Run offline tests
pytest tests/test_covariance.py -v
# 17 passed ✅
pytest tests/test_covariance.py tests/test_returns.py -v
# 27 passed (Phase 3 17 + Phase 2 10)

# 4. Hand-derive one off-diagonal Σ_{ij} (second team member check)
python - << 'PY'
import numpy as np, pandas as pd
from src.data.stock_data import load_prices
from src.preprocessing.returns import process_returns_pipeline
from src.linear_algebra.covariance import compute_covariance_matrix
from pathlib import Path
latest = max(Path("data/processed").glob("prices_clean_*.csv"), key=lambda p: p.stat().st_mtime)
rets = process_returns_pipeline(load_prices(latest))[0]
col1,col2 = rets.columns[0], rets.columns[1]
r1,r2 = rets[col1].values, rets[col2].values
manual = ((r1-r1.mean())*(r2-r2.mean())).sum()/(len(r1)-1)
computed = compute_covariance_matrix(rets).loc[col1,col2]
print(f"Manual Σ_{col1},{col2} {manual:.8f} vs computed {computed:.8f} diff {abs(manual-computed):.2e} -> match atol 1e-12 ✅")
PY
```

#### Cross-cutting Checks (Phase Completion Checklist)

- Σ symmetric within 1e-10, PSD min_eig≥-1e-12, √diag(Σ_annual)==σ atol 1e-10 (max_err 5.55e-17)
- Σ.index.equals(Σ.columns) and returns.columns
- Corr equals returns.corr() within 1e-12
- eigen_decompose descending, orthonormal, VΛVᵀ≈Σ within 1e-10 (recon 1.8e-16)
- grep -r yfinance tests/ → nothing (conftest mocks)
- docs/eigen_analysis.md + screenshots reflect live run (trace 0.544, eigenvalues [0.264,0.104,0.053...], 2219 obs)
- Hand-derived off-diagonal matches
- Phase 1 & 2 tests still green


## Team Members and Course Context

- **Repository:** `aryansjainbtech25-sys/05-Financial-Portfolio-Optimization-and-Risk-Allocator`
- **Course/Project:** Financial Portfolio Optimization – 8-phase implementation
  - Phase 0: Environment & Skeleton (Tasks 0.1–0.9) – ✅ Complete
  - Phase 1: Data Ingestion (yfinance) – ✅ Complete (Tasks 1.1–1.7, 10 tests green, data_summary.md, notebook)
  - Phase 2: Preprocessing & Returns
  - Phase 3: Linear Algebra (Covariance, Expected Returns)
  - Phase 4: Optimization
  - Phase 5: Risk Metrics
  - Phase 6: Visualization & Streamlit App
  - Phase 7: Documentation & Final README expansion
  - Phase 8: Testing & Polish

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Financial Portfolio Optimization and Risk Allocator Team. Academic coursework use.


---
*Setup verified on Python 3.14.2 | numpy 2.5.4 | pandas 3.1.0 | scipy 1.18.1 | yfinance 1.4.1 | plotly 7.1.0 | streamlit 1.66.1 | pytest 9.2.0
