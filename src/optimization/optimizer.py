"""
src/optimization/optimizer.py
Phase 4 — Tasks 4.6, 4.7, 4.8: Portfolio optimizers.

This module hosts the three Markowitz optimizers. Each is a thin wrapper
around scipy.optimize.minimize(method="SLSQP") with a shared feasible
set (fully invested, long-only) and a shared return-dict contract.

Specification contract (see docs/optimization_specification.md):
    Section 2.1:   min_w  wᵀ Σ w   s.t. 1ᵀw = 1, 0 ≤ w ≤ 1
    Section 2.2:   max_w (wᵀμ - r_f)/√(wᵀΣw) s.t. same constraints
    Section 2.3:   min_w wᵀ Σ w    s.t. wᵀμ = target, 1ᵀw = 1, 0 ≤ w ≤ 1
    Section 3:     SLSQP, tol=1e-10, maxiter=1000, ftol=1e-12
    Section 5.3:   return dict keys: weights, return, volatility,
                   variance, sharpe, success, message, n_iter.
                   Downgrade success if post-validation fails.
    Section 5.4:   if scipy raises before producing x, return
                   weights=None, nan metrics, success=False.
    Section 5.5:   reject non-finite inputs on entry.
    Section 6 (Q6): r_f is annualized decimal (0.02 = 2%/yr).

At the end of Phase 4 Task 4.6, this module exposes only
`minimum_variance_portfolio`. Tasks 4.7 and 4.8 will add
`maximum_sharpe_portfolio` and `target_return_portfolio` respectively.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize

from src.optimization.benchmarks import equal_weights
from src.optimization.constraints import (
    budget_constraint,
    long_only_bounds,
    validate_weights,
    target_return_constraint
)
from src.optimization.portfolio_metrics import (
    portfolio_return,
    portfolio_sharpe,
    portfolio_variance,
    portfolio_volatility,
)

__all__ = [
    "minimum_variance_portfolio",
    "maximum_sharpe_portfolio",
    "target_return_portfolio",
]

# ---------------------------------------------------------------------------
# Constants (spec Section 3)
# ---------------------------------------------------------------------------

_SLSQP_TOL = 1e-10
_SLSQP_OPTIONS = {"maxiter": 1000, "ftol": 1e-12}

# Sentinel objective value used when the Sharpe ratio is undefined
# (zero-volatility weights). 1e10 is far above any plausible negative
# Sharpe magnitude on annualized data, so SLSQP steers away from the
# singularity (spec Section 5.2 / Q4).
_SHARPE_ZERO_VOL_SENTINEL = 1e10

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_inputs(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: Optional[float],
) -> tuple[pd.Index, np.ndarray, np.ndarray, Optional[float]]:
    """
    Validate and convert inputs into numpy arrays plus a stable label index.

    Raises ValueError on non-finite inputs or shape mismatch.
    Raises AssertionError on label mismatch between mu and Sigma.
    """
    # mu
    if isinstance(mu, pd.Series):
        mu_idx = mu.index
        mu_arr = mu.to_numpy(dtype=float, copy=False)
    else:
        mu_idx = None
        mu_arr = np.asarray(mu, dtype=float)

    if mu_arr.ndim != 1:
        raise ValueError(f"mu must be 1-D, got shape {mu_arr.shape}")
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("mu contains non-finite values")

    # Sigma
    if isinstance(Sigma, pd.DataFrame):
        sigma_idx = Sigma.index
        if not Sigma.index.equals(Sigma.columns):
            raise ValueError("Sigma index and columns do not match")
        Sigma_arr = Sigma.to_numpy(dtype=float, copy=False)
    else:
        sigma_idx = None
        Sigma_arr = np.asarray(Sigma, dtype=float)

    if Sigma_arr.ndim != 2 or Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise ValueError(f"Sigma must be square 2-D, got shape {Sigma_arr.shape}")
    if Sigma_arr.shape[0] != mu_arr.shape[0]:
        raise ValueError(
            f"dimension mismatch: mu has {mu_arr.shape[0]} entries, "
            f"Sigma has shape {Sigma_arr.shape}"
        )
    if not np.all(np.isfinite(Sigma_arr)):
        raise ValueError("Sigma contains non-finite values")

    # Label alignment
    if mu_idx is not None and sigma_idx is not None and not mu_idx.equals(sigma_idx):
        raise AssertionError(
            f"Label mismatch: mu.index={list(mu_idx)!r}, "
            f"Sigma.index={list(sigma_idx)!r}"
        )
    labels = mu_idx if mu_idx is not None else sigma_idx
    if labels is None:
        raise ValueError(
            "at least one of mu / Sigma must carry ticker labels "
            "(pass as pd.Series / pd.DataFrame)"
        )

    # Risk-free rate
    if risk_free_rate is not None:
        if not np.isfinite(risk_free_rate):
            raise ValueError(f"risk_free_rate must be finite, got {risk_free_rate}")
        r_f: Optional[float] = float(risk_free_rate)
    else:
        r_f = None

    return labels, mu_arr, Sigma_arr, r_f


def _build_failure_result(
    exc: BaseException,
    labels: pd.Index,
) -> dict:
    """
    Return dict for the spec Section 5.4 pre-solver-exception case:
    scipy.optimize.minimize raised before producing any x.
    """
    n = len(labels)
    return {
        "weights": None,
        "return": float("nan"),
        "volatility": float("nan"),
        "variance": float("nan"),
        "sharpe": float("nan"),
        "success": False,
        "message": f"solver raised {type(exc).__name__}: {exc}",
        "n_iter": 0,
        "n_assets": n,  # extra info, harmless
    }


def _pack_result(
    x: np.ndarray,
    labels: pd.Index,
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    r_f: Optional[float],
    success: bool,
    message: str,
    n_iter: int,
) -> dict:
    """
    Assemble the standard return dict from a solver output vector x.

    Steps:
    1. Wrap x in a labeled pd.Series.
    2. Recompute all metrics from the wrapped weights (do NOT trust
       scipy's result.fun — it may differ slightly).
    3. Post-validate weights. If solver said success=True but weights
       fail validation, downgrade success=False per spec Section 5.3.
    4. If r_f is None, sharpe is None (spec Section 5.3).
    """
    w_series = pd.Series(x, index=labels)

    # Recompute metrics from the wrapped weights
    r_p = portfolio_return(w_series, mu)
    var_p = portfolio_variance(w_series, Sigma)
    vol_p = portfolio_volatility(w_series, Sigma)

    if r_f is None:
        sharpe_val: Optional[float] = None
    else:
        try:
            sharpe_val = portfolio_sharpe(w_series, mu, Sigma, r_f)
        except ZeroDivisionError:
            # Degenerate weights — treat as validation failure below.
            sharpe_val = float("nan")

    # Post-validate per spec Section 5.3
    try:
        validate_weights(w_series, labels=labels)
        final_success = success
        final_message = message
    except ValueError as exc:
        final_success = False
        final_message = f"solver reported success but weights failed validation: {exc}"

    return {
        "weights": w_series,
        "return": float(r_p),
        "volatility": float(vol_p),
        "variance": float(var_p),
        "sharpe": (float(sharpe_val) if sharpe_val is not None else None),
        "success": final_success,
        "message": final_message,
        "n_iter": int(n_iter),
    }


# ---------------------------------------------------------------------------
# Task 4.6 — minimum-variance optimizer
# ---------------------------------------------------------------------------


def minimum_variance_portfolio(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: Optional[float] = None,
) -> dict:
    """
    Find the portfolio that minimizes σ_p² = wᵀ Σ w
    subject to 1ᵀw = 1 and 0 ≤ w_i ≤ 1.

    Args:
        mu:             (N,) expected returns (annualized)
        Sigma:          (N,N) covariance matrix (annualized)
        risk_free_rate: annualized decimal, or None.
                        If provided, the Sharpe ratio is computed for
                        the optimal portfolio. If None, sharpe=None.

    Returns:
        dict with keys (spec Section 5.3):
            weights:    pd.Series, ticker-indexed
            return:     float
            volatility: float
            variance:   float
            sharpe:     float | None
            success:    bool
            message:    str
            n_iter:     int

    Raises:
        ValueError: on non-finite inputs, shape mismatch, or missing labels.
        AssertionError: on label mismatch between mu and Sigma.
    """
    labels, mu_arr, Sigma_arr, r_f = _prepare_inputs(mu, Sigma, risk_free_rate)
    n = len(labels)

    # Initial guess: equal weights (spec Section 3)
    w0 = equal_weights(labels).to_numpy()

    # Feasible set: budget + long-only (spec Section 3)
    constraints = [budget_constraint()]
    bounds = long_only_bounds(n)

    # Objective: σ_p² = wᵀ Σ w, routed through the Task 4.3 primitive.
    # Using portfolio_variance (not an inline w @ Sigma @ w) is a design
    # invariant: the test suite proves the routing.
    def objective(w: np.ndarray) -> float:
        return portfolio_variance(w, Sigma_arr)

    try:
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=_SLSQP_TOL,
            options=_SLSQP_OPTIONS,
        )
    except Exception as exc:
        return _build_failure_result(exc, labels)

    if not res.success:
        return {
            "weights": pd.Series(res.x, index=labels) if res.x is not None else None,
            "return": float("nan"),
            "volatility": float("nan"),
            "variance": float("nan"),
            "sharpe": float("nan") if r_f is not None else None,
            "success": False,
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", 0)),
        }

    return _pack_result(
        x=res.x,
        labels=labels,
        mu=mu_arr,
        Sigma=Sigma_arr,
        r_f=r_f,
        success=True,
        message="Minimum-variance optimization converged",
        n_iter=getattr(res, "nit", 0),
    )

# ---------------------------------------------------------------------------
# Task 4.7 — maximum-Sharpe optimizer
# ---------------------------------------------------------------------------


def maximum_sharpe_portfolio(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: float,
) -> dict:
    """
    Find the portfolio that maximizes the Sharpe ratio
        S_p = (wᵀμ − r_f) / √(wᵀΣw)
    subject to 1ᵀw = 1 and 0 ≤ w_i ≤ 1.

    Args:
        mu:             (N,) expected returns (annualized)
        Sigma:          (N,N) covariance matrix (annualized)
        risk_free_rate: annualized decimal (0.02 = 2%/yr). REQUIRED —
                        a Sharpe ratio is undefined without r_f.

    Returns:
        dict with the standard Phase 4 keys (spec Section 5.3).
        If the maximum achievable Sharpe is negative (no long-only
        portfolio beats r_f), success is still True and `sharpe` is
        negative; the message explains the situation (spec Section 5.1, Q3).

    Raises:
        ValueError: on non-finite inputs, shape mismatch, or missing labels.
        AssertionError: on label mismatch between mu and Sigma.
        TypeError: if risk_free_rate is omitted (Python raises this
                   automatically because there is no default).
    """
    labels, mu_arr, Sigma_arr, r_f = _prepare_inputs(mu, Sigma, risk_free_rate)
    assert r_f is not None  # guaranteed by the required-argument signature
    n = len(labels)

    w0 = equal_weights(labels).to_numpy()
    constraints = [budget_constraint()]
    bounds = long_only_bounds(n)

    def objective(w: np.ndarray) -> float:
        """
        Negative Sharpe, with ZeroDivisionError caught and mapped to the
        sentinel 1e10 (spec Section 5.2 / Q4). Routes through the Task 4.3
        portfolio_sharpe primitive — do not inline the ratio.
        """
        try:
            sharpe = portfolio_sharpe(w, mu_arr, Sigma_arr, r_f)
        except ZeroDivisionError:
            return _SHARPE_ZERO_VOL_SENTINEL
        return -float(sharpe)

    try:
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=_SLSQP_TOL,
            options=_SLSQP_OPTIONS,
        )
    except Exception as exc:
        return _build_failure_result(exc, labels)

    if not res.success:
        return {
            "weights": pd.Series(res.x, index=labels) if res.x is not None else None,
            "return": float("nan"),
            "volatility": float("nan"),
            "variance": float("nan"),
            "sharpe": float("nan"),
            "success": False,
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", 0)),
        }

    result = _pack_result(
        x=res.x,
        labels=labels,
        mu=mu_arr,
        Sigma=Sigma_arr,
        r_f=r_f,
        success=True,
        message="Maximum-Sharpe optimization converged",
        n_iter=getattr(res, "nit", 0),
    )

    # Q3 (spec Section 5.1): if the true maximum is negative, keep
    # success=True but explain the situation in the message.
    if result["sharpe"] is not None and result["sharpe"] < 0:
        result["message"] = (
            f"Maximum-Sharpe optimization converged. All long-only "
            f"portfolios have negative Sharpe at r_f = {r_f}; returning "
            f"the maximum even though it is negative "
            f"(sharpe = {result['sharpe']:.6f})."
        )
    return result

# ---------------------------------------------------------------------------
# Task 4.8 — target-return min-variance optimizer
# ---------------------------------------------------------------------------


def target_return_portfolio(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    target_return: float,
    risk_free_rate: Optional[float] = None,
) -> dict:
    """
    Find the minimum-variance portfolio subject to a target expected return:
        min_w  wᵀ Σ w
        s.t.   wᵀ μ = target_return
               1ᵀ w = 1
               0 ≤ w_i ≤ 1

    This is the frontier-building primitive Phase 5 will call in a loop
    to trace the efficient frontier.

    Args:
        mu:             (N,) expected returns (annualized)
        Sigma:          (N,N) covariance matrix (annualized)
        target_return:  scalar target portfolio return (annualized decimal).
                        Must satisfy mu.min() ≤ target ≤ mu.max(); outside
                        this range the solver returns success=False per Q1/Q2.
        risk_free_rate: annualized decimal, or None. If provided, Sharpe
                        is computed for the resulting portfolio; if None,
                        sharpe is None.

    Returns:
        Standard Phase 4 dict (spec Section 5.3). The achieved return is
        in `result["return"]`. If the solver succeeds but the achieved
        return differs from target by more than 1e-6, success is
        downgraded per spec Section 4.3.

    Raises:
        ValueError: on non-finite inputs, shape mismatch, or missing labels.
        AssertionError: on label mismatch between mu and Sigma.
    """
    labels, mu_arr, Sigma_arr, r_f = _prepare_inputs(mu, Sigma, risk_free_rate)
    n = len(labels)

    # Q1/Q2 range check — fail loudly BEFORE calling scipy (spec 4.1, 4.2).
    if not np.isfinite(target_return):
        raise ValueError(f"target_return must be finite, got {target_return}")

    mu_min = float(mu_arr.min())
    mu_max = float(mu_arr.max())
    if target_return < mu_min:
        return {
            "weights": None,
            "return": float("nan"),
            "volatility": float("nan"),
            "variance": float("nan"),
            "sharpe": float("nan") if r_f is not None else None,
            "success": False,
            "message": (
                f"target_return = {target_return:.6f} is below mu.min() = "
                f"{mu_min:.6f}; infeasible under long-only constraint"
            ),
            "n_iter": 0,
        }
    if target_return > mu_max:
        return {
            "weights": None,
            "return": float("nan"),
            "volatility": float("nan"),
            "variance": float("nan"),
            "sharpe": float("nan") if r_f is not None else None,
            "success": False,
            "message": (
                f"target_return = {target_return:.6f} is above mu.max() = "
                f"{mu_max:.6f}; infeasible under long-only constraint"
            ),
            "n_iter": 0,
        }

    w0 = equal_weights(labels).to_numpy()
    constraints = [
        budget_constraint(),
        target_return_constraint(mu_arr, float(target_return)),
    ]
    bounds = long_only_bounds(n)

    def objective(w: np.ndarray) -> float:
        return portfolio_variance(w, Sigma_arr)

    try:
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=_SLSQP_TOL,
            options=_SLSQP_OPTIONS,
        )
    except Exception as exc:
        return _build_failure_result(exc, labels)

    if not res.success:
        return {
            "weights": pd.Series(res.x, index=labels) if res.x is not None else None,
            "return": float("nan"),
            "volatility": float("nan"),
            "variance": float("nan"),
            "sharpe": float("nan") if r_f is not None else None,
            "success": False,
            "message": str(res.message),
            "n_iter": int(getattr(res, "nit", 0)),
        }

    result = _pack_result(
        x=res.x,
        labels=labels,
        mu=mu_arr,
        Sigma=Sigma_arr,
        r_f=r_f,
        success=True,
        message="Target-return optimization converged",
        n_iter=getattr(res, "nit", 0),
    )

    # Spec Section 4.3: post-check that the achieved return is on target.
    if result["success"]:
        achieved = result["return"]
        diff = abs(achieved - float(target_return))
        if diff > 1e-6:
            result["success"] = False
            result["message"] = (
                f"solver reported success but achieved return {achieved:.10f} "
                f"differs from target {target_return:.10f} by {diff:.3e} "
                f"(tolerance 1e-6)"
            )
    return result