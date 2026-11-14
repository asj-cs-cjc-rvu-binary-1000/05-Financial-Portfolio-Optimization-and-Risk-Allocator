# Phase 6 — Interactive Dashboard Specification

**Status:** Approved
**Date:** 2026-11-14
**Applies to:** Tasks 6.3 – 6.8 (scaffold, config, optimization, visualization, error handling, verification)
**Consumers:** Phase 7 (final tests, screenshots, presentation)

This document is the single source of truth for the Streamlit dashboard's
architecture, controls, layout, caching, and error-handling. Any test in
`tests/test_dashboard.py` (Task 6.8) and any behavior in `app/app.py` must be
consistent with this specification. If code and this document disagree, the
document wins and the code is a bug.

---

## 1. Architecture Overview

- **One file:** `app/app.py`. A single Streamlit entry point.
- **One page with three tabs:** `Optimization`, `Risk Analysis`, `Frontier`.
- **All controls in the sidebar** (`st.sidebar`). Main panel is 100% results.
- **No multi-page routing.** The three tabs share the sidebar state and see
  the same loaded data / `mu` / `Sigma` / cached frontier analysis.
- **No new mathematical code.** Every quantity displayed is the output of a
  Phase 4 or Phase 5 public function. The dashboard is a *rendering layer*.

### 1.1 Directory layout

```
app/
├── __init__.py
└── app.py                 # entry point (main())
```

Optionally, if the app grows beyond ~400 lines, split into:

```
app/
├── __init__.py
├── app.py                 # entry point (main())
└── components/
    ├── __init__.py
    ├── sidebar.py         # render_sidebar() -> DashboardInputs
    ├── optimization_tab.py# render_optimization_tab(inputs, data)
    ├── risk_tab.py        # render_risk_tab(inputs, data)
    └── frontier_tab.py    # render_frontier_tab(inputs, data)
```

The single-file `app.py` is the **default** for Phase 6. Only split into
`components/` if the file exceeds 400 lines. Task 6.3 makes this decision
concrete at implementation time.

---

## 2. Dashboard Layout

### 2.1 Sidebar controls (top to bottom)

| Control | Widget | Default | Notes |
| :--- | :--- | :--- | :--- |
| Ticker selection | `st.multiselect` | `DEFAULT_TICKERS` (all 7) | From the CSV's columns. **Must select ≥ 2.** |
| Date range | `st.date_input` (range) | Full CSV range | Filters prices **before** recomputing `mu` / `sigma` / `Sigma` — see §2.1a. |
| Risk-free rate (%) | `st.slider` (0.0–20.0%, step 0.5%) | 2.0% | Converted to decimal at boundary. `st.warning` if > 15%. |
| VaR/CVaR confidence | `st.selectbox` | 95% | Options: 90%, 95%, 99%. |
| Frontier resolution | `st.slider` (20–100, step 10) | 50 | Passed to `analyze_frontier(n_points=…)`. |
| Include Target Return | `st.checkbox` | False | When False, no target slider and only 3 rows in tables. |
| Target return (%) | `st.slider` (dynamic bounds) | `mu.mean()` | Only shown if checkbox is True. Bounds: `mu.min()` → `mu.max()` of the *currently selected* assets. |
| Run button | `st.button` | — | Disabled if any pre-run validation fails. |

### 2.1a Date-range filtering and recomputation

When the user narrows the date range, annualized `mu` and `Sigma` **cannot
be sliced** from the full-history values — annualized means and covariances
are not additive across sub-ranges. The dashboard therefore:

1. Slices the raw price DataFrame to `[date_start, date_end]` and the
   selected tickers.
2. Re-runs `process_returns_pipeline` on the slice.
3. Recomputes `mu`, `sigma`, and `Sigma` from the sliced returns.

This is implemented in `app/app.py::_filtered_data` (cached). The cost is
one extra `process_returns_pipeline` call per unique slice — cheap
(~50ms), fully cached, and numerically correct.

### 2.2 Pre-run validation rules (Q7 refined)

The "Run" button is disabled (or shows `st.error`) when any of these hold:

| Rule | Message |
| :--- | :--- |
| `< 2` tickers selected | "Select at least 2 tickers to compute covariance." |
| `date_start >= date_end` | "Start date must be before end date." |
| `risk_free_rate` not finite | "Risk-free rate must be a finite number." |
| Target-return enabled but target outside `[mu.min(), mu.max()]` | "Target must be between X% and Y% for the selected assets." |

**Non-blocking warnings** (do not disable the Run button):

| Rule | Message |
| :--- | :--- |
| `risk_free_rate > 0.15` | "Risk-free rate above 15% is unusual — check the input." |

**Explicitly out of scope:** no fixed upper or lower bound on `r_f` beyond
"finite and numeric". The Phase 4 specification defines `r_f` as an
annualized decimal but does not bound it. Do not impose `[0, 0.20]`.

### 2.3 Main content — Tab 1 (`Optimization`)

**Sections, top to bottom:**

1. **Summary cards** — four small metric cards in a row:
   - Number of assets selected
   - Risk-free rate (percent)
   - Confidence level (percent)
   - Target return (percent) if enabled, else "—"

2. **Comparison table** — output of `compare_portfolios(mu, Sigma, r_f, target)`,
   displayed via `st.dataframe`. All numeric columns formatted as per §3.

3. **Allocation chart** — grouped bar chart of weights (one bar per strategy).
   Colors match: EqualWeight `#888888`, MinVolatility `#2ca02c`,
   MaxSharpe `#d62728`, TargetReturn `#9467bd`.

4. **Success/failure panel** — one `st.warning` per failed optimizer row,
   showing its `message` field verbatim. If all succeed, a single
   `st.success("All portfolios solved.")`.

### 2.4 Main content — Tab 2 (`Risk Analysis`)

1. **Risk comparison table** — `compare_risk_metrics(mu, Sigma, returns, r_f, target, confidence)`.
   Columns: return, volatility, sharpe, var_95, cvar_95, max_drawdown,
   success, one per ticker. Same color coding as Tab 1.

2. **Portfolio selector** — `st.radio` to pick which portfolio to inspect
   ("EqualWeight", "MinVolatility", "MaxSharpe", ["TargetReturn"]). Default:
   the portfolio with the highest Sharpe among successful rows.

3. **Drawdown chart** — `plot_drawdown(drawdown_series(portfolio_returns), portfolio_name)`.

4. **VaR histogram** — `plot_var_histogram(portfolio_returns, var_value, portfolio_name, confidence)`.

5. **Risk metric cards** — three small cards: VaR (as %), CVaR (as %),
   Max Drawdown (as %). Formatted per §3.

### 2.5 Main content — Tab 3 (`Frontier`)

1. **Configuration reminder** — small text: "Frontier computed at r_f = X%,
   n_points = Y, confidence = Z%."

2. **Efficient frontier chart** — `plot_efficient_frontier(analysis)`, with
   `analysis = analyze_frontier(mu, Sigma, r_f, n_points)` (cached).

3. **Anchor card row** — three cards side by side:
   - Min Volatility: return, volatility, Sharpe
   - Max Sharpe: return, volatility, Sharpe
   - Equal Weight: return, volatility, Sharpe

4. **Metadata table** — the `metadata` dict from `analyze_frontier`
   (`mu_min`, `mu_min_var`, `mu_max`, `n_success_efficient`, `n_success_inefficient`,
   `n_failed_efficient`, `n_failed_inefficient`) displayed as a 2-column
   key–value table.

### 2.6 Loading indicators

All expensive calls (`process_returns_pipeline`, `analyze_frontier`) are
wrapped with `with st.spinner("Computing…"):`. The spinner is shown only
on a cache miss.

### 2.7 Empty / error states

| Condition | Behavior |
| :--- | :--- |
| `data/processed/` is empty | Show `st.error("No processed prices found. Run `python -m src.data.stock_data` first.")`. Do not attempt `load_prices`. |
| CSV load raises | Show `st.error(f"Failed to load data: {e}")`. Do not render tabs. |
| `process_returns_pipeline` raises | Same — show error + `return` from main. |
| One optimizer row has `success=False` | Show the row in the table (with `NaN` cells) + `st.warning` with the row's `message`. Other rows still render. |
| All optimizer rows fail | Show three `st.warning` panels; skip the allocation chart. |
| `analyze_frontier` raises | Show `st.error(f"Frontier failed: {e}")` in Tab 3 only. Tabs 1 and 2 still work. |

---

## 3. Formatting Conventions (Q4 refined)

| Quantity | Format | Example |
| :--- | :--- | :--- |
| Return (annualized) | `f"{x*100:.2f}%"` | `"19.88%"` |
| Volatility (annualized) | `f"{x*100:.2f}%"` | `"18.43%"` |
| Variance | `f"{x:.4f}"` | `"0.0340"` |
| Sharpe ratio | `f"{x:.3f}"` | `"0.970"` |
| VaR / CVaR | `f"{x*100:.2f}%"` | `"1.66%"`, `"2.77%"` |
| Max drawdown | `f"{x*100:.2f}%"` (preserve minus) | `"-30.19%"` |
| Weights | `f"{x:.4f}"` | `"0.1429"` |
| Confidence level | `f"{int(x*100)}%"` | `"95%"` |
| Risk-free rate | `f"{x*100:.1f}%"` | `"2.0%"` |
| `n_iter` | `str(x)` | `"25"` |
| Boolean `success` | emoji checkmark / cross | `"✅"` / `"❌"` |

**Rule:** never change the underlying `float` before calling a Phase 4/5
function. Format only at display time. `st.dataframe` can display the raw
DataFrame with `column_config` / `NumberColumn(format=…)`, but for metric
cards we format explicitly with `f-strings`.

---

## 4. Caching (Q6 refined)

Four `@st.cache_data` boundaries, keyed on their arguments:

```python
@st.cache_data(show_spinner=False)
def _load_data(path: str) -> pd.DataFrame: ...

@st.cache_data(show_spinner=False)
def _returns_pipeline(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]: ...

@st.cache_data(show_spinner=False)
def _covariance(returns: pd.DataFrame) -> pd.DataFrame: ...

@st.cache_data(show_spinner=False)
def _analyze_frontier(mu: pd.Series, Sigma: pd.DataFrame, r_f: float, n_points: int) -> dict: ...

**Notes:**

- `st.cache_data` serializes returns via pickle; `pd.DataFrame` and `pd.Series` are picklable.
- Two calls with numerically equal but distinct `mu` / `Sigma` objects will hit the same cache slot. This is desired.
- Sliders/selectboxes trigger a rerun on change, so cache invalidation is automatic.
- **Never** use `@st.cache_resource` for these — it is intended for connections, not DataFrames.
- The four functions above are `app-private` (leading underscore), not exported from any `src/` module.

---

## 5. Error Handling (Q7 refined — three layers)

### 5.1 Layer 1 — Pre-run validation

See §2.2. Runs before any expensive computation. Blocks the Run button (or
shows `st.error`) on invalid input. **Does not use `st.stop()`** — the
sidebar and empty tabs still render.

### 5.2 Layer 2 — Runtime exceptions

Each tab's rendering is a function (`render_optimization_tab`,
`render_risk_tab`, `render_frontier_tab`). Each is wrapped in a
`try/except` at the top-level loop:

```python
for tab_obj, render_fn, name in zip(tabs, [render_optimization_tab, render_risk_tab, render_frontier_tab], ["Optimization", "Risk", "Frontier"]):
    with tab_obj:
        try:
            render_fn(inputs, data)
        except Exception as e:
            st.error(f"[{name}] {type(e).__name__}: {e}")
            # Do not st.stop(); other tabs still need to render.
```

`st.stop()` is **never** used. A failure in Tab 3 does not prevent Tabs 1
and 2 from rendering.

### 5.3 Layer 3 — Post-run failure inspection

Every Phase 4/5 function returns either a `dict` with a `success: bool`
key (optimizers) or a `pd.DataFrame` with a `success` column (comparison
functions). The dashboard inspects these after each call:

- If `success is False` → `st.warning(f"…: {result['message']}")`.
- If `success is True` → render normally.

No false-positive "success" check: the `success` field is the source of truth.

---

## 6. Q8 — Target-Return UI

- Default state: checkbox **unchecked**.
- When checked, a slider appears with dynamic bounds `[mu.min(), mu.max()]`
  computed from the currently selected assets' `mu`.
- The slider's `value` initializes to `mu.mean()` on each selection change.
- If the user checks the box, moves the slider, then unchecks the box, the
  target is not passed to `compare_portfolios` / `compare_risk_metrics`.
- `target_return_portfolio` may return `success=False` with a message
  containing "below mu.min" or "above mu.max" (Q1/Q2 range failures from
  Phase 4 spec §4.1–4.2). The dashboard catches this and shows
  `st.warning(result["message"])` — no crash.

---

## 7. Chart Reuse (Q5 refined)

The dashboard **only** calls the three Phase 5 chart functions:

- `plot_efficient_frontier(analysis)` — Tab 3.
- `plot_drawdown(drawdown_series, portfolio_name)` — Tab 2.
- `plot_var_histogram(returns, var_value, portfolio_name, confidence_level)` — Tab 2.

All three return `go.Figure`. Rendered with:

```python
st.plotly_chart(fig, use_container_width=True)
```

**Do not** reimplement any of them inside `app.py`. Do not use
`st.line_chart` / `st.bar_chart` for these. Do not use `matplotlib` /
`seaborn`.

For the **allocation bar chart** (Task 6.5, Tab 1 §2.3), the dashboard
builds a small `go.Bar` figure inline. This is the only chart the dashboard
constructs itself, because it is a dashboard-specific composition (weight
bars across strategies) — not a Phase 5 deliverable.

---

## 8. Launch

**How to run the dashboard:**

```bash
source .venv/bin/activate
streamlit run app/app.py
# Streamlit serves on http://localhost:8501 (forwarded by devcontainer)
```

The devcontainer already forwards port 8501 (Phase 0, Task 0.5). The
Codespace will show the forwarded port in the Ports tab.

**How to stop:** Ctrl+C in the terminal.

**Requirements:** `data/processed/prices_clean_*.csv` must exist. If
missing, run `python -m src.data.stock_data` first. The dashboard does
**not** download data itself.

---

## 9. Testing Strategy (Task 6.8)

**Test file:** `tests/test_dashboard.py`.

Streamlit apps are hard to test via Selenium, and the project has no
browser automation. The tests will be **unit tests on the pure helper
functions** in `app/app.py` (or `app/components/`):

- `_format_return(x)`, `_format_pct(x)`, `_format_sharpe(x)` — pure formatters.
- `_validate_inputs(inputs) -> list[str]` — returns a list of error messages.
- `_target_slider_bounds(mu) -> (lo, hi)` — pure function.

Every function that does not touch `st.*` is testable. Functions that
emit widgets (`st.multiselect`, `st.slider`) are not unit-tested — they are
visually verified.

**Estimated tests:** ~8–12.

**Real-data verification (Task 6.8 acceptance criteria):** run the
dashboard manually, note a portfolio's VaR/MDD/Sharpe from the UI, then
call the underlying function directly and confirm they match to
`1e-6`. This is a manual check, not a pytest.

---

## 10. Non-Goals (out of scope for Phase 6)

- Multi-page routing (`st.Page`, `st.navigation`) — the three-tab layout covers the brief.
- Live yfinance fetches inside Streamlit — data must come from the CSV.
- User accounts, authentication, database storage.
- Export to PDF / Excel / CSV (nice-to-have; skip unless trivial).
- Theme customization beyond `st.set_page_config(layout="wide")`.
- Streamlit `session_state`-based undo / history.
- Selenium / Playwright end-to-end tests.

---

## 11. Approval Record

| Q | Choice | Rationale |
|---|---|---|
| Q1 Single vs multi-page | A | Single page, three tabs. Simplest correct UX. |
| Q2 Sidebar | A | All controls in sidebar. Idiomatic Streamlit. |
| Q3 Asset selection | A | Multiselect from CSV, N ≥ 2 enforced. No network calls in dashboard. |
| Q4 Formatting | A | Percent for returns/volatility/VaR/CVaR/MDD; decimal Sharpe; decimal weights. Format at display time only. |
| Q5 Charts | A | Reuse Phase 5 chart functions verbatim; no new chart code. |
| Q6 Caching | A | `@st.cache_data` on four pipeline boundaries. |
| Q7 Errors | A (refined) | Three layers: pre-run validation, runtime exception catch per tab, post-run success-flag inspection. **No `st.stop()`.** Risk-free rate finite-and-numeric only; no hard bound. |
| Q8 Target return | A | Optional checkbox; slider bounded dynamically by selected assets' `mu.min()/mu.max()`. |

Approved by: project team (Bibek and Aryan).
Approving supervisor: Senior Software Engineer, Linear Algebra Project Mentor.

---

## 12. Grader References (Phase 6, forward-looking)

- Task 6.1: `docs/phase6_inputs_verification.md` — verified Phase 5 baseline, 234 tests
- Task 6.2: this document
- Task 6.3: `app/app.py` — Streamlit scaffold; verified `streamlit run app/app.py` launches
- Task 6.4: interactive config — sidebar controls match §2.1
- Task 6.5: `compare_portfolios` integration; results identical to CLI (`python -m src.optimization.optimization_pipeline`)
- Task 6.6: efficient frontier, drawdown, VaR histogram integrated
- Task 6.7: three-layer error handling; no `st.stop()`; graceful failure modes
- Task 6.8: `tests/test_dashboard.py`; real-data verification
