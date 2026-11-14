# Phase 5 — Risk Analysis Specification

**Status:** Approved
**Date:** 2026-11-14
**Applies to:** Tasks 5.3 – 5.8 (risk metrics, frontier generation, visualization, integration, documentation)
**Consumers:** Phase 6 (Streamlit dashboard), Phase 7 (final tests + presentation)

This document is the single source of truth for Phase 5's risk metrics,
efficient-frontier generation, and visualization conventions. Any test in
`tests/test_risk_metrics.py`, `tests/test_frontier.py`, and downstream phases
must be consistent with this specification. If code and this document
disagree, the document wins and the code is a bug.

---

## 1. Notation

| Symbol | Meaning | Shape | Type |
|---|---|---|---|
| `returns` | Daily simple returns, indexed by date | (T, N) | `pd.DataFrame` |
| `w` | Fixed portfolio weight vector | (N,) | `pd.Series` or `np.ndarray` |
| `R_p,t` | Portfolio return on day t: `R_p,t = w^T r_t` | scalar | `float` |
| `r_f` | Annualized risk-free rate (decimal) | scalar | `float` |
| `c` | VaR / CVaR confidence level (decimal) | scalar | `float` |
| `Q_p(R)` | Empirical p-quantile of return series R | scalar | `float` |

Returns are **daily simple** (see Phase 2 `RETURN_TYPE = "simple"`). Risk metrics
operate on **daily** return series (the `returns` DataFrame columns). Do not
annualize inside a risk metric — that is a display-time concern for Phase 6.

---

## 2. Risk Metrics

### 2.1 Maximum drawdown (Q1)

**Contract:** `maximum_drawdown(returns) -> float` returns a **signed value ≤ 0**.

Companion: `drawdown_series(returns) -> pd.Series` returns the full
peak-to-trough drawdown series, same DatetimeIndex as `returns`.

**Formulas:**

$$
C_t = \prod_{s=1}^{t} (1 + r_s), \quad
P_t = \max_{s \le t} C_s, \quad
D_t = \frac{C_t - P_t}{P_t}
$$

$$
\text{MDD} = \min_t D_t \le 0
$$

**Identity:** `maximum_drawdown(returns) == drawdown_series(returns).min()`.
This identity is a test invariant and must never be broken.

**Why signed:** portfolio returns are signed quantities in this project
(`−0.05` means lost 5%). MDD answers "what was the worst cumulative path?"
and its signed answer is `−0.25`, not `+0.25`. This keeps MDD consistent
with the drawdown series and the return convention.

**What about VaR's opposite sign convention?** See Section 2.3.

### 2.2 Historical VaR (Q2, Q3, Q4)

**Contract:** `historical_var(returns, confidence_level=0.95) -> float`

**Formula:**

$$
\text{VaR}_c = -Q_{1-c}(R)
$$

where `Q_{1-c}` is the empirical `1-c` quantile of the daily return series.
Default `c = 0.95`, giving the 5% quantile.

**Conventions:**

- **Horizon:** daily. Do not apply `√252` scaling inside the metric. Annualized
  VaR is a display-time concern for Phase 6.
- **Sign:** positive loss magnitude (industry convention). A return of `+0.02`
  from `historical_var` means "in 95% of days, the loss does not exceed 2%."
- **Positive-quantile edge case:** if the 5% quantile of returns is itself
  positive (unusual — some asset gained money on its worst 5% of days),
  `historical_var` returns a **negative** number, signaling "no VaR, the tail
  is positive." Documented test case.

**Confidence level (Q2):** default `0.95`, overridable via parameter. Phase 6
dashboard may expose this; the metric function should accept any `0 < c < 1`.
Out-of-range `c` raises `ValueError`.

### 2.3 Historical CVaR / Expected Shortfall (Q5)

**Contract:** `historical_cvar(returns, confidence_level=0.95) -> float`

**Formula:**

$$
\text{CVaR}_c = -\mathbb{E}\!\left[R \mid R \le Q_{1-c}(R)\right]
$$

i.e., the **negative of the mean** of all returns at or below the VaR threshold.

**Conventions:** identical to VaR (daily, positive loss, `c=0.95` default).
CVaR ≤ VaR in magnitude (the average bad day is worse than the threshold day),
which is a test invariant.

**Edge case:** if fewer than 2 observations lie below the threshold, CVaR
equals VaR and a warning is emitted — this happens with very short return
series. Documented test case.

### 2.4 Sign convention summary

| Function | Returns | Typical value | Sign | Reasoning |
|---|---|---|---|---|
| `maximum_drawdown` | float | `−0.25` | **signed negative** | Consistent with `drawdown_series().min()`; MDD is a realized return |
| `drawdown_series` | `pd.Series` | `[... −0.25 ...]` | **signed negative** | Peak-to-trough cumulative return, by definition ≤ 0 |
| `historical_var` | float | `0.02` | **positive loss** | Industry convention: "2% VaR" means "at most 2% daily loss on 95% of days" |
| `historical_cvar` | float | `0.03` | **positive loss** | Same convention as VaR; CVaR ≥ VaR in magnitude |

**The asymmetry is intentional.** VaR and CVaR report a *threshold* (loss magnitude
convention, as every risk desk does); MDD reports a *realized cumulative return*
(signed convention, as every return series is). Both are correct. Phase 6 is
the layer that formats these for display (e.g., "−25% peak-to-trough" and
"2% daily 95%").

---

## 3. Portfolio Historical Returns (Q6)

**Contract:** `portfolio_historical_returns(w, returns) -> pd.Series`

$$
R_{p,t} = w^\top r_t \quad \text{for each day } t
$$

**Documented assumptions:**

- **Fixed target weights.** `w` is the optimal weight vector from Phase 4
  (min-variance, max-Sharpe, target-return, or equal-weight). It does not change
  over the historical window.
- **Daily rebalancing implicit.** At each day `t`, the portfolio is presumed to
  have been rebalanced to `w` at the start of the day. This means the portfolio
  return on day `t` is exactly `w^T r_t`, with no weight drift from price moves.
- **No transaction costs.** Rebalancing is free.
- **No rolling re-optimization.** We do not re-run `minimum_variance_portfolio` or
  `maximum_sharpe_portfolio` at any intermediate date.
- **Not a historical trading backtest.** This is a *static allocation analysis* —
  it answers "what would my drawdown/VaR have been if I had held these fixed
  weights over the whole history?" It is not a simulation of a real trading
  strategy with transaction costs, tax, or capital gains.

These assumptions must be documented in the docstring of any function that
computes portfolio-level risk metrics.

---

## 4. Efficient Frontier (Q7)

**Contract:**

```python
efficient_frontier(mu, Sigma, n_points=50) -> tuple[pd.DataFrame, pd.DataFrame, dict]
```

Returns `(efficient_df, inefficient_df, metadata)`.

### 4.1 `efficient_df`

- **Interval:** `[mu_min_var, mu_max]` — the efficient upper branch.
- **Points:** `n_points` evenly-spaced target returns (default 50).
- **Columns:** `target_return`, `return`, `volatility`, `variance`, `success`, `n_iter`, plus one column per ticker (weights).
- **Rows:** one per target. Failed optimizations appear as `success=False` with `NaN` metrics and `NaN` weights (do not drop — the caller decides).
- **Endpoint behavior:** `target = mu_min_var` produces the minimum-variance portfolio. `target = mu_max` produces the corner solution (all-in on the highest-μ asset, if long-only permits).

### 4.2 `inefficient_df`

- **Interval:** `[mu_min, mu_min_var]` — the lower branch (inefficient, dominated).
- **Points:** `n_points` evenly-spaced target returns.
- **Columns / rows:** identical schema to `efficient_df`.

### 4.3 `metadata`

Dict with keys:

- `mu_min` — `mu.min()` (lowest single-asset expected return)
- `mu_min_var` — expected return of the minimum-variance portfolio (from `minimum_variance_portfolio`)
- `mu_max` — `mu.max()` (highest single-asset expected return)
- `n_success_efficient` — count of `success=True` rows in `efficient_df`
- `n_success_inefficient` — count of `success=True` rows in `inefficient_df`
- `n_failed_efficient` — count of `success=False` rows in `efficient_df`
- `n_failed_inefficient` — count of `success=False` rows in `inefficient_df`
- `n_points` — the requested `n_points` (echo)

Both branches call `target_return_portfolio` from Phase 4 Task 4.8. The
`mu_min_var` endpoint is shared (both branches include it).

### 4.4 Determinism

Same `(mu, Sigma, n_points)` → byte-identical DataFrames on the same environment.
Cross-machine determinism of SLSQP is not guaranteed; a tolerance of `1e-8` on
`target_return` achieved values is acceptable and should not be treated as a bug.

---

## 5. Visualizations (Q8)

Task 5.5/5.6 produce a `go.Figure` per chart (no `.show()` calls inside — the
caller decides whether to display, save to PNG, or hand to Streamlit).

### 5.1 Efficient frontier plot

**Layers (in this order, so later layers draw on top):**

1. **Inefficient branch** — dashed gray line (`color="gray", dash="dash"`).
   Plotted first so it does not obscure the efficient branch.
2. **Efficient branch** — solid blue line (`color="#1f77b4"`).
3. **Individual assets** — scatter points `(σ_i, μ_i)` for each ticker, labeled with the ticker symbol in gray.
4. **Minimum-volatility portfolio** — labeled point, green triangle marker, annotation "Min Vol".
5. **Maximum-Sharpe portfolio** — labeled point, red star marker, annotation "Max Sharpe".
6. **Equal-weight portfolio** — labeled point, gray circle marker, annotation "Equal Weight".

**Axes:** x = annualized volatility, y = annualized expected return.
**Title:** "Efficient Frontier — Markowitz Mean-Variance".
**Legend:** shows all six layers; the asset scatter is one entry labeled "Individual Assets".

### 5.2 Drawdown chart

- **Series:** `drawdown_series` of the selected portfolio, plotted as a filled area chart.
- **X-axis:** dates.
- **Y-axis:** drawdown as a **percentage** (converted from decimal at display time — `drawdown * 100`).
- **Title:** "Drawdown — `<portfolio name>`".

### 5.3 Historical VaR histogram

- **Series:** daily returns of the selected portfolio, plotted as a histogram.
- **Threshold line:** a vertical red line at `−VaR_c` (i.e., at the `5%` quantile, `Q_{1-c}`). Note: the line is at the quantile *in return space*, not the VaR value.
- **Tail shading:** returns at or below the threshold are shaded (via a second histogram trace or a `vrect` shape).
- **Title:** "Daily Returns with VaR Threshold — `<portfolio name>` (95%)".

### 5.4 No non-Plotly dependencies

All charts use `plotly.graph_objects` and `plotly.express`. **No `matplotlib`, `seaborn`.**

---

## 6. Non-Goals (out of scope for Phase 5)

- Rolling-rebalanced backtest with transaction costs.
- Factor-model risk decomposition beyond eigen-decomposition (which lives in Phase 3).
- Parametric (normal) VaR — this phase is historical only.
- Annualized VaR inside the metric function.
- Capital gains tax modeling.
- Time-varying covariance.

---

## 7. Approval Record

| Q | Choice | Rationale |
|---|---|---|
| Q1 Max drawdown | A + series | Cumulative simple-return MDD, signed negative, companion `drawdown_series` for the chart. |
| Q2 VaR confidence | A + C | Default 0.95, overridable via parameter. |
| Q3 VaR horizon | A | Daily. No `√252` scaling inside the metric. |
| Q4 VaR sign | A | Positive loss magnitude. Negative if 5% quantile is positive. |
| Q5 CVaR | A | Included. Same convention as VaR. |
| Q6 Portfolio risk | A | Fixed weights, daily rebalancing, no transaction costs, not a backtest. |
| Q7 Frontier | A + extension | Two branches (efficient + inefficient), 50 points each, metadata dict. |
| Q8 Visualization | C | Both branches, assets as scatter, three labeled portfolio points. |

Approved by: project team (Bibek and Aryan).
Approving supervisor: Senior Software Engineer, Linear Algebra Project Mentor.