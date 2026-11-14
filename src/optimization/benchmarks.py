"""
src/optimization/benchmarks.py
Phase 4 — Task 4.5: Equal-weight benchmark portfolio.

The equal-weight portfolio w_eq = (1/N) · 1 is the reference every
optimizer in Phase 4 is measured against. It is computed analytically
(no scipy.optimize call), but it is returned in the EXACT same dict
shape as the three optimizers so that the comparison pipeline (Task 4.9)
and the Phase 6 dashboard can iterate over all portfolios uniformly.

Return-dict contract (matches Tasks 4.6, 4.7, 4.8 exactly):
    weights      pd.Series (N,), ticker-indexed
    return       float    (annualized, w @ mu)
    volatility   float    (annualized, sqrt(w @ Sigma @ w))
    variance     float    (annualized, w @ Sigma @ w)
    sharpe       float or None  (None only when risk_free_rate is None)
    success      bool     (always True for this benchmark)
    message      str      ("Equal-weight benchmark (analytical)")
    n_iter       int      (0 — no solver ran)

Specification: see docs/optimization_specification.md Section 2.4.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from src.optimization.constraints import validate_weights
from src.optimization.portfolio_metrics import (
    portfolio_return,
    portfolio_sharpe,
    portfolio_variance,
    portfolio_volatility,
)

__all__ = [
    "equal_weights",
    "equal_weight_portfolio",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_labels(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
) -> pd.Index:
    """
    Derive the ticker labels for the weight vector. Prefers mu.index;
    falls back to Sigma.index if mu is an ndarray. Raises if neither
    source provides labels (equal-weight needs stable labels for the
    returned pd.Series).
    """
    mu_idx = mu.index if isinstance(mu, pd.Series) else None
    sigma_idx = Sigma.index if isinstance(Sigma, pd.DataFrame) else None

    if mu_idx is not None and sigma_idx is not None:
        if not mu_idx.equals(sigma_idx):
            raise AssertionError(
                f"Label mismatch: mu.index={list(mu_idx)!r}, "
                f"Sigma.index={list(sigma_idx)!r}"
            )
        return mu_idx
    if mu_idx is not None:
        return mu_idx
    if sigma_idx is not None:
        return sigma_idx

    raise ValueError(
        "equal_weight_portfolio: neither mu nor Sigma carries ticker labels; "
        "cannot construct a labeled pd.Series. Pass at least one as "
        "pd.Series / pd.DataFrame."
    )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def equal_weights(labels: Union[list, pd.Index, pd.Series]) -> pd.Series:
    """
    Build the equal-weight vector 1/N · 1 for the given ticker labels.

    Pure builder — no metrics, no validation. Used by Task 4.6–4.8 as
    the initial guess w0 = 1/N · 1 for scipy.optimize.minimize.

    Args:
        labels: iterable of ticker names (list), pd.Index, or pd.Series
                (whose .index becomes the label source).

    Returns:
        pd.Series of length N, values 1/N, index = labels.

    Raises:
        ValueError: if labels is empty.
    """
    if isinstance(labels, pd.Series):
        idx = labels.index
    elif isinstance(labels, pd.Index):
        idx = labels
    else:
        idx = pd.Index(list(labels))

    n = len(idx)
    if n == 0:
        raise ValueError("equal_weights: labels is empty")
    return pd.Series([1.0 / n] * n, index=idx)


def equal_weight_portfolio(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: Optional[float] = None,
) -> dict:
    """
    Return the equal-weight benchmark portfolio with full metrics.

    Args:
        mu:             (N,) expected returns (annualized), Series or ndarray
        Sigma:          (N,N) covariance matrix (annualized), DataFrame or ndarray
        risk_free_rate: annualized decimal (0.02 = 2%/yr), or None.

    Returns:
        dict with the exact shape the optimizers in Tasks 4.6–4.8 will return:
            weights:    pd.Series
            return:     float
            volatility: float
            variance:   float
            sharpe:     float | None
            success:    bool
            message:    str
            n_iter:     int
    """
    labels = _extract_labels(mu, Sigma)
    w_eq = equal_weights(labels)

    # Design invariant: the benchmark itself must be feasible.
    # If this ever fails, constraints.py or the caller has a bug.
    validate_weights(w_eq, labels=labels)

    r_p = portfolio_return(w_eq, mu)
    var_p = portfolio_variance(w_eq, Sigma)
    vol_p = portfolio_volatility(w_eq, Sigma)

    if risk_free_rate is None:
        sharpe_val: Optional[float] = None
    else:
        sharpe_val = portfolio_sharpe(w_eq, mu, Sigma, risk_free_rate)

    return {
        "weights": w_eq,
        "return": float(r_p),
        "volatility": float(vol_p),
        "variance": float(var_p),
        "sharpe": (float(sharpe_val) if sharpe_val is not None else None),
        "success": True,
        "message": "Equal-weight benchmark (analytical)",
        "n_iter": 0,
    }