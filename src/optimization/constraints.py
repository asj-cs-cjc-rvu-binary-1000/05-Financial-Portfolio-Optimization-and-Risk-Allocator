"""
src/optimization/constraints.py
Phase 4 — Task 4.4: Portfolio weight validation and constraint helpers.

All optimizers in Phase 4 (Tasks 4.6–4.8) share the same feasible set:

    Fully invested:  1ᵀ w = 1
    Long-only:       0 ≤ w_i ≤ 1  for each asset i

This module defines the four helpers that every optimizer must call.
Constraint semantics live here and nowhere else, so they cannot
silently diverge between minimum-variance, maximum-Sharpe, and
target-return optimizers.

Specification contract (see docs/optimization_specification.md):
- Section 3 (Q7): fully invested + long-only, no weight caps,
  w0 = equal weights, scipy SLSQP.
- Section 5.3: every optimizer must post-validate weights via
  `validate_weights` before returning `success=True`.

scipy.optimize.minimize convention:
    {"type": "eq", "fun": f}  means  f(w) == 0
    {"type": "ineq", "fun": f} means  f(w) >= 0
Bounds are passed as list[tuple[float, float]], not as constraints.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "validate_weights",
    "budget_constraint",
    "long_only_bounds",
    "target_return_constraint",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOL = 1e-6
"""Tolerance for weight validation. Matches spec Section 5.3."""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def validate_weights(
    w: Union[pd.Series, np.ndarray],
    labels: Optional[pd.Index] = None,
    tol: float = DEFAULT_TOL,
) -> bool:
    """
    Validate that w is a feasible fully-invested, long-only portfolio.

    Checks:
        1. w is 1-D and non-empty.
        2. If `labels` is provided and w is a pd.Series, w.index must
           equal labels exactly (same values, same order).
        3. All weights are finite (no NaN, no inf).
        4. |sum(w) - 1| <= tol.
        5. Every w_i is in [-tol, 1 + tol].

    Args:
        w:       (N,) weight vector.
        labels:  optional ticker index; if provided, w.index is
                 checked against it. Ignored if w is a plain ndarray.
        tol:     absolute tolerance for sum and bounds.

    Returns:
        True if valid.

    Raises:
        ValueError: if any check fails, with a descriptive message.
    """
    # Shape and finiteness
    if isinstance(w, pd.Series):
        if labels is not None and not w.index.equals(labels):
            raise ValueError(
                f"validate_weights: w.index does not match labels. "
                f"w.index={list(w.index)!r}, labels={list(labels)!r}"
            )
        w_arr = w.to_numpy(dtype=float, copy=False)
    else:
        w_arr = np.asarray(w, dtype=float)

    if w_arr.ndim != 1:
        raise ValueError(f"validate_weights: w must be 1-D, got shape {w_arr.shape}")
    if w_arr.size == 0:
        raise ValueError("validate_weights: w is empty")
    if not np.all(np.isfinite(w_arr)):
        raise ValueError("validate_weights: w contains non-finite values")

    # Budget
    s = float(w_arr.sum())
    if abs(s - 1.0) > tol:
        raise ValueError(
            f"validate_weights: sum(w) = {s:.10f}, "
            f"differs from 1 by more than tol={tol}"
        )

    # Long-only bounds
    w_min = float(w_arr.min())
    w_max = float(w_arr.max())
    if w_min < -tol:
        raise ValueError(
            f"validate_weights: negative weight detected "
            f"(min = {w_min:.6e}, tol = {tol:.1e})"
        )
    if w_max > 1.0 + tol:
        raise ValueError(
            f"validate_weights: weight exceeds 1 "
            f"(max = {w_max:.6e}, tol = {tol:.1e})"
        )

    return True


def budget_constraint() -> dict:
    """
    Return the equality constraint 1ᵀ w = 1 in scipy.optimize.minimize format.

    The returned dict is {"type": "eq", "fun": f}, where f(w) = sum(w) - 1.
    scipy interprets "eq" as f(w) == 0, so this is exactly the budget constraint.
    """
    return {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}


def long_only_bounds(n_assets: int) -> list[tuple[float, float]]:
    """
    Return [(0.0, 1.0)] * n_assets — the scipy bounds for a long-only,
    fully-invested portfolio. Any weight outside [0, 1] is forbidden.

    Args:
        n_assets: number of assets (must be >= 1).

    Returns:
        List of (min, max) tuples of length n_assets.

    Raises:
        ValueError: if n_assets < 1.
    """
    if not isinstance(n_assets, int) or n_assets < 1:
        raise ValueError(f"long_only_bounds: n_assets must be int >= 1, got {n_assets!r}")
    return [(0.0, 1.0)] * n_assets


def target_return_constraint(
    mu_vec: Union[pd.Series, np.ndarray],
    target: float,
) -> dict:
    """
    Return the equality constraint wᵀμ = target in scipy format.

    f(w) = w @ mu_vec - target.  scipy reads "eq" as f(w) == 0, so
    any w returned by the optimizer satisfies w @ mu_vec == target
    to within the solver tolerance.

    IMPORTANT — ticker ordering: the returned fun assumes that `w` is
    passed in the SAME order as `mu_vec`. Callers must construct `mu_vec`
    to match the weight vector's assumed order (typically derived from
    `Sigma.index`, which is identical to `mu.index` per Task 4.1). If
    w is a pd.Series whose index differs from mu_vec's index, this
    function does not detect that — it works on raw numpy values.

    Args:
        mu_vec: (N,) expected returns, in the same order as the weight
                vector that scipy will pass to fun.
        target: scalar target portfolio return (annualized decimal).

    Returns:
        {"type": "eq", "fun": f} suitable for scipy.optimize.minimize.
    """
    if isinstance(mu_vec, pd.Series):
        mu_arr = mu_vec.to_numpy(dtype=float, copy=False)
    else:
        mu_arr = np.asarray(mu_vec, dtype=float)

    if mu_arr.ndim != 1:
        raise ValueError(
            f"target_return_constraint: mu_vec must be 1-D, got shape {mu_arr.shape}"
        )
    if not np.all(np.isfinite(mu_arr)):
        raise ValueError("target_return_constraint: mu_vec contains non-finite values")
    if not np.isfinite(target):
        raise ValueError(f"target_return_constraint: target must be finite, got {target}")

    target_f = float(target)

    def _fun(w: np.ndarray) -> float:
        w_arr = np.asarray(w, dtype=float)
        if w_arr.shape != mu_arr.shape:
            raise ValueError(
                f"target_return_constraint: w shape {w_arr.shape} != "
                f"mu_vec shape {mu_arr.shape}"
            )
        return float(w_arr @ mu_arr - target_f)

    return {"type": "eq", "fun": _fun}