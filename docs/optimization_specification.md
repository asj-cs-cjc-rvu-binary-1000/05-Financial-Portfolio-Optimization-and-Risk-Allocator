# Phase 4 — Portfolio Optimization Specification

**Status:** Approved
**Date:** 2026-11-13
**Applies to:** Tasks 4.3 – 4.10 (portfolio metrics, optimizers, comparison pipeline, CLI)
**Consumers:** Phase 5 (efficient frontier), Phase 6 (Streamlit dashboard), Phase 7 (final tests)

This document is the single source of truth for the optimization engine's
objectives, constraints, edge-case behavior, and numerical conventions. Any
test in `tests/test_optimizer.py`, `tests/test_portfolio_metrics.py`, and
downstream phases must be consistent with this specification. If code and this
document disagree, the document wins and the code is a bug.

---

## 1. Notation

| Symbol | Meaning | Shape | Type |
|---|---|---|---|
| `mu` (μ) | Annualized expected return vector | (N,) | `pd.Series`, ticker-indexed |
| `Sigma` (Σ) | Annualized covariance matrix | (N,N) | `pd.DataFrame`, ticker-indexed |
| `w` | Portfolio weight vector | (N,) | `pd.Series` or `np.ndarray` |
| `r_f` | Annualized risk-free rate (decimal) | scalar | `float` |
| `R_p` | Portfolio expected return, `w @ mu` | scalar | `float` |
| `sigma_p^2` | Portfolio variance, `w @ Sigma @ w` | scalar | `float` |
| `sigma_p` | Portfolio volatility, `sqrt(sigma_p^2)` | scalar | `float` |
| `S_p` | Sharpe ratio, `(R_p - r_f) / sigma_p` | scalar | `float` |
| `N` | Number of assets (tickers) | scalar | `int` |

The ticker order of `mu`, the row order of `Sigma`, and the column order of
`Sigma` are identical and are provided by Phase 2/Phase 3. All optimizer
functions assert this invariant on entry and raise `AssertionError` if violated.

---

## 2. Objective Functions

### 2.1 Minimum-variance portfolio (Task 4.6)

**Objective:** minimize `w @ Sigma @ w`

**Notation:**
    argmin_w  wᵀ Σ w
    s.t.      1ᵀ w = 1
              0 ≤ w_i ≤ 1  for all i

### 2.2 Maximum-Sharpe portfolio (Task 4.7)

**Objective:** maximize `(w @ mu - r_f) / sqrt(w @ Sigma @ w)`,
implemented as minimizing the negative:

    argmin_w  − (wᵀ μ − r_f) / sqrt(wᵀ Σ w)
    s.t.      1ᵀ w = 1
              0 ≤ w_i ≤ 1

### 2.3 Target-return portfolio (Task 4.8)

**Objective:** minimize variance subject to a fixed expected return:

    argmin_w  wᵀ Σ w
    s.t.      wᵀ μ = target
              1ᵀ w = 1
              0 ≤ w_i ≤ 1

### 2.4 Equal-weight benchmark (Task 4.5)

    w_eq = (1/N) · 1

Computed analytically, not via the optimizer. Used only for comparison.

---

## 3. Constraints and Bounds (Q7)

All optimizers enforce:

- **Fully invested:** `1ᵀ w = 1`, implemented as an equality constraint
  `{"type": "eq", "fun": lambda w: w.sum() - 1.0}`.
- **Long-only, no leverage, no shorts:** `0 ≤ w_i ≤ 1` for every asset,
  implemented as per-asset bounds `[(0.0, 1.0)] * N`.
- **Initial guess:** `w0 = (1/N) · 1` — the equal-weight portfolio. Not zeros,
  not random. Rationale: SLSQP is a local solver; `w0 = 0` gives a
  zero-gradient at the boundary, and random points make results
  non-deterministic. Equal weights start strictly interior.
- **No optimizer-level maximum or minimum weight caps** (Q7 Option A).
  The brief requires `0 ≤ w_i ≤ 1` only. If a downstream consumer (e.g.,
  the Phase 6 dashboard) wants to display a "clean" allocation with tiny
  weights dropped, that is a *display-time* transformation and must not be
  applied inside the optimizer.

Solver: `scipy.optimize.minimize`, `method="SLSQP"`,
`tol=1e-10`, `options={"maxiter": 1000, "ftol": 1e-12}`.

---

## 4. Target-Return Feasibility (Q1, Q2)

### 4.1 Lower bound (Q1 = A)

A target is feasible in principle if `target >= mu.min()`.

- If `target < mu.min()`, `target_return_portfolio` returns
  `success=False` with a message naming `mu.min()` and the requested target.
  It does **not** silently clip.
- The min-variance point (`mu_min_var`) is *not* the lower bound. The
  min-variance envelope extends below it into the inefficient frontier, and
  Phase 5 is responsible for clipping its sweep to the efficient side only.

### 4.2 Upper bound (Q2 = A)

A target is feasible in principle if `target <= mu.max()`.

- At exactly `target == mu.max()`, the solution is the corner portfolio
  `w = e_i` where `i` is the highest-μ asset. This is a valid long-only
  solution and the optimizer is allowed to return it.
- If `target > mu.max()`, return `success=False` with a clear message.
- No artificial epsilon is subtracted from `mu.max()`.

### 4.3 Numerical target tolerance

When a solver reports `success=True`, the achieved return must satisfy

    |wᵀ μ − target| ≤ 1e-6

otherwise the result is downgraded to `success=False` with a message
"achieved return X differs from target Y by more than 1e-6".

---

## 5. Edge-Case Behavior

### 5.1 Negative Sharpe maximum (Q3 = A)

If every long-only portfolio has a negative Sharpe ratio at the supplied
`r_f`, the optimizer still returns the candidate with the **highest** (least
negative) Sharpe. The returned dict has:

- `success = True`
- `sharpe` = the negative maximum
- `message` explaining "all long-only portfolios have negative Sharpe at
  r_f = X; returning the maximum even though it is negative"

The optimizer must not:
- return `success=False`,
- substitute the equal-weight portfolio,
- clip Sharpe to zero.

Phase 6 (dashboard) is responsible for displaying a user-visible warning
when `result["sharpe"] < 0` regardless of `success`.

### 5.2 Zero-volatility Sharpe (Q4 = A)

Public function `portfolio_sharpe(w, mu, Sigma, r_f)`:

- computes `sigma_p = portfolio_std(w, Sigma)`
- if `sigma_p < 1e-12`, raises `ZeroDivisionError` with a message that
  names `w` and `sigma_p`.

Internal max-Sharpe objective function (inside Task 4.7):

- tries `portfolio_sharpe(w, mu, Sigma, r_f)`
- catches `ZeroDivisionError` and returns the sentinel objective value
  `1e10` (a very bad value for a minimizer, so SLSQP steers away).

Sentinel `1e10` is chosen well above any plausible negative Sharpe
magnitude (annualized Sharpe in this project's data is O(1)), so it cannot
accidentally look like the best candidate.

### 5.3 Solver failure (Q5 = A)

Every optimizer returns a dict with all of the following keys:

- `weights` — `pd.Series`, ticker-indexed (or `None` if solver failed
  *before* producing any `x`) — see 5.4
- `return` — float, `w @ mu`
- `volatility` — float, `sqrt(w @ Sigma @ w)`
- `variance` — float, `w @ Sigma @ w`
- `sharpe` — float or `None` (None only when `r_f` was not supplied
  and the optimizer is min-variance / target-return)
- `success` — bool
- `message` — str, human-readable, from `result.message` or post-validation
- `n_iter` — int, `result.nit` (or 0 if solver did not run)

**Behavior when `result.success == False`:**

- Return the dict with `success=False`.
- Still expose the final (unvalidated) `weights` if the solver produced
  an `x` — the caller may want to diagnose the failure.
- Set `message` to `result.message`.
- Do **not** raise.

**Behavior when `result.success == True` but post-validation fails:**

- Compute `is_valid_weights(result.x)` (sum within `1e-6`, all in `[0, 1]`
  within `1e-6`).
- If invalid, downgrade `success=False` and set `message` to
  "solver reported success but weights failed validation: <reason>".
- Do not silently truncate or renormalize the weights.

### 5.4 Solver failure before any `x` exists

If `scipy.optimize.minimize` raises (rare, but possible for badly scaled
inputs), each optimizer catches the exception and returns the same
dict shape with:

- `weights = None`
- `return = volatility = variance = float('nan')`
- `sharpe = float('nan')` (or `None` when `r_f` is not applicable)
- `success = False`
- `message = f"solver raised {type(exc).__name__}: {exc}"`
- `n_iter = 0`

Callers must be prepared for `weights is None` when `success=False`.
A successful result **always** has non-None weights.

### 5.5 Non-finite inputs

Every public function in `src/optimization/` validates that `mu` and `Sigma`
contain no NaN or infinite values on entry and raises `ValueError` if they
do. Silent NaN propagation through the objective function is a class of bug
this project explicitly forbids.

---

## 6. Risk-Free Rate Convention (Q6 = A)

`r_f` is a **scalar float, annualized, in decimal form**.

| User intent | Value passed |
|---|---|
| 0% per year | `0.0` |
| 2% per year | `0.02` |
| 5% per year | `0.05` |

This matches the annualization convention of `mu` (annualized mean) and
`Sigma` (annualized covariance):

- `mu_annual = mu_daily × 252`
- `Sigma_annual = Sigma_daily × 252`
- `r_f` is provided **already annualized** — no internal multiplication
  by 252 occurs inside `portfolio_sharpe` or any optimizer.

Phase 6 (dashboard) is responsible for converting a user-entered percent
(e.g., `2` for 2%) to decimal (`0.02`) at the boundary. The optimization
engine itself never accepts percent input.

---

## 7. Return-Value Contract

Every optimizer function returns a plain Python `dict` with **exactly** the
keys listed in 5.3 (extra keys, e.g., `target_return` for the target-return
optimizer, are allowed; missing keys are a contract violation).

Downstream consumers (Phase 5, Phase 6, Phase 7 tests) access the dict
using `result["weights"]`, `result["return"]`, `result["volatility"]`,
`result["sharpe"]`, `result["success"]`. Adding a key is a minor change;
removing one is a breaking change.

---

## 8. Determinism

For a given `(mu, Sigma, r_f)` triple and fixed `w0 = (1/N) · 1`:

- Every optimizer produces the **same** numeric output across runs
  on the same machine and library version.
- Cross-machine determinism is not guaranteed for SLSQP; this is why
  `docs/optimization_summary.md` (Task 4.6) records the environment
  versions alongside the table.

A test that compares outputs must compare against numbers derived from the
SAME environment; a test that compares two runs on the SAME environment
must use exact equality or a very tight tolerance (`1e-10`).

---

## 9. Non-Goals (out of scope for Phase 4)

The following are explicitly **not** part of this phase. If they appear in
a "Phase 4" PR, they are scope creep:

- Short positions (`w_i < 0`), leverage (`1ᵀ |w| > 1`), or margin.
- Individual weight caps (`w_i ≤ cap`) or floors (`w_i ≥ floor`).
- Transaction costs or turnover penalties.
- Risk parity optimization.
- Efficient-frontier *tracing* (that is Phase 5; Phase 4 provides the
  target-return primitive that Phase 5 calls in a loop).
- Dollar-amount allocation (that is Phase 6; the optimization engine
  works in fractional weights only).

---

## 10. Approval Record

| Decision | Choice | Rationale |
|---|---|---|
| Q1 target-return lower bound | A — `mu.min()` | Mechanical feasibility; efficient-frontier clipping is Phase 5's job |
| Q2 target-return upper bound | A — `mu.max()`, corner allowed | Consistent with Q1; corner solution is a valid long-only answer |
| Q3 negative Sharpe | A — return the true max, `success=True`, warn in message | Honest mathematical answer; dashboard is the warning layer |
| Q4 zero-volatility Sharpe | A — raise in public fn, sentinel `1e10` in objective | Prevents `inf`/`nan` gradients inside SLSQP |
| Q5 solver failure | A — `success=False`, expose weights, distinguish pre/post-validation | SciPy convention; dashboard can diagnose |
| Q6 risk-free rate | A — annualized decimal | Consistent with `mu`, `Sigma`; one conversion in Phase 6 only |
| Q7 constraints | A — `1ᵀw=1`, `0≤w_i≤1`, no weight caps, `w0=1/N·1` | Matches the project brief exactly |

Approved by: project team (Bibek and Aryan).
Approving supervisor: Senior Software Engineer, Linear Algebra Project Mentor.