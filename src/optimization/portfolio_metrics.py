"""
src/optimization/portfolio_metrics.py
Phase 4 — Task 4.3: Portfolio metric calculations

Four scalar functions that are the mathematical core of the optimization
engine. Every optimizer (Tasks 4.6, 4.7, 4.8) must call these — never
re-implement `w @ mu` or `w @ Sigma @ w` inline.

Specification contract (see docs/optimization_specification.md):
- Section 2.1, 2.2, 2.3: these metrics define the optimizers' objectives.
- Section 5.2 (Q4): `portfolio_sharpe` raises `ZeroDivisionError` when
  `portfolio_volatility < 1e-12`. The internal max-Sharpe objective
  (Task 4.7) catches this and returns sentinel 1e10.
- Section 5.5: non-finite `mu` / `Sigma` raise `ValueError` on entry.
- Section 6 (Q6): `risk_free_rate` is an annualized decimal (0.02 = 2%/yr).

Mathematical definitions:
    R_p      = wᵀ μ
    σ_p²     = wᵀ Σ w    (quadratic form, routes through quadratic_form)
    σ_p      = √(wᵀ Σ w)
    S_p      = (R_p − r_f) / σ_p

Shape contract:
    w      : (N,)     pd.Series or np.ndarray
    mu     : (N,)     pd.Series or np.ndarray
    Sigma  : (N,N)    pd.DataFrame or np.ndarray (symmetric PSD)
    return : float (Python scalar, not numpy scalar)

All label mismatches between w, mu, Sigma are rejected with AssertionError.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from src.linear_algebra.matrix_ops import quadratic_form

__all__ = [
    "portfolio_return",
    "portfolio_variance",
    "portfolio_volatility",
    "portfolio_sharpe",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VOLATILITY_ZERO_TOL = 1e-12
# Volatility below this value is treated as zero for Sharpe purposes.
# Chosen in Task 4.2 Q4: raises ZeroDivisionError at or below this threshold.

_NONFINITE_TOL = 1e-12  # unused placeholder for future tightening


def _as_1d_array(x: Union[pd.Series, np.ndarray], name: str) -> np.ndarray:
    """Convert w or mu to a 1-D float ndarray. Raise on non-1D or non-finite."""
    if isinstance(x, pd.Series):
        arr = x.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_2d_array(M: Union[pd.DataFrame, np.ndarray], name: str) -> np.ndarray:
    """Convert Sigma to a 2-D float ndarray. Raise on non-2D or non-finite."""
    if isinstance(M, pd.DataFrame):
        arr = M.to_numpy(dtype=float, copy=False)
    else:
        arr = np.asarray(M, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _assert_label_alignment(
    w: Union[pd.Series, np.ndarray],
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
) -> None:
    """
    If any of w, mu, Sigma carry ticker labels, all labeled inputs must
    agree on both the labels and their order. Silent reindexing is
    forbidden — order mismatches are bugs.
    """
    w_idx = w.index if isinstance(w, pd.Series) else None
    mu_idx = mu.index if isinstance(mu, pd.Series) else None
    sigma_rows = Sigma.index if isinstance(Sigma, pd.DataFrame) else None
    sigma_cols = Sigma.columns if isinstance(Sigma, pd.DataFrame) else None

    label_sources = [("w", w_idx), ("mu", mu_idx),
                     ("Sigma.index", sigma_rows), ("Sigma.columns", sigma_cols)]
    present = [(name, idx) for name, idx in label_sources if idx is not None]

    if len(present) < 2:
        return  # no cross-check possible

    ref_name, ref_idx = present[0]
    for name, idx in present[1:]:
        if not ref_idx.equals(idx):
            raise AssertionError(
                f"Label mismatch: {ref_name} vs {name}. "
                f"{ref_name}={list(ref_idx)!r}, {name}={list(idx)!r}"
            )


# ---------------------------------------------------------------------------
# Public metrics
# ---------------------------------------------------------------------------


def portfolio_return(
    w: Union[pd.Series, np.ndarray],
    mu: Union[pd.Series, np.ndarray],
) -> float:
    """
    Expected portfolio return R_p = wᵀ μ.

    Args:
        w:  (N,) weights
        mu: (N,) expected returns (annualized)

    Returns:
        Python float. Shape in: (N,), (N,) -> out: scalar float.
    """
    _assert_label_alignment(w, mu, np.zeros((0, 0)))  # only w vs mu possible
    w_arr = _as_1d_array(w, "w")
    mu_arr = _as_1d_array(mu, "mu")
    if w_arr.shape[0] != mu_arr.shape[0]:
        raise ValueError(
            f"dimension mismatch: w has {w_arr.shape[0]} entries, "
            f"mu has {mu_arr.shape[0]}"
        )
    return float(w_arr @ mu_arr)


def portfolio_variance(
    w: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
) -> float:
    """
    Portfolio variance σ_p² = wᵀ Σ w, computed via the canonical
    `quadratic_form` primitive from Phase 3.

    This is a design-invariant test: monkeypatching `quadratic_form` to
    raise must cause this function to raise (test 15 in Task 4.8).

    Args:
        w:     (N,) weights
        Sigma: (N,N) symmetric PSD covariance matrix

    Returns:
        Python float. Shape in: (N,), (N,N) -> out: scalar float.
    """
    _assert_label_alignment(w, np.zeros_like(np.asarray(w)) if not isinstance(w, pd.Series) else w,
                            Sigma)
    w_arr = _as_1d_array(w, "w")
    Sigma_arr = _as_2d_array(Sigma, "Sigma")
    if Sigma_arr.shape[0] != Sigma_arr.shape[1]:
        raise ValueError(f"Sigma must be square, got shape {Sigma_arr.shape}")
    if Sigma_arr.shape[0] != w_arr.shape[0]:
        raise ValueError(
            f"dimension mismatch: w has {w_arr.shape[0]} entries, "
            f"Sigma has shape {Sigma_arr.shape}"
        )
    # Route through canonical primitive. Do NOT inline w @ Sigma @ w.
    return quadratic_form(w_arr, Sigma_arr)


def portfolio_volatility(
    w: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
) -> float:
    """
    Portfolio volatility σ_p = √(wᵀ Σ w).

    Guards against tiny negative round-off by clamping sqrt input at 0,
    but raises RuntimeError if raw variance is more negative than -1e-12
    (that indicates a genuine bug — non-PSD Sigma or a Phase 3 regression).

    Args:
        w:     (N,) weights
        Sigma: (N,N) covariance matrix

    Returns:
        Python float ≥ 0. Shape in: (N,), (N,N) -> out: scalar float.
    """
    var = portfolio_variance(w, Sigma)
    if var < -1e-12:
        raise RuntimeError(
            f"portfolio_variance returned a large negative value ({var}); "
            "Sigma may not be positive semi-definite"
        )
    return float(np.sqrt(max(0.0, var)))


def portfolio_sharpe(
    w: Union[pd.Series, np.ndarray],
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: float,
) -> float:
    """
    Sharpe ratio S_p = (R_p − r_f) / σ_p.

    Q4 contract: raises ZeroDivisionError when σ_p < 1e-12 (a genuinely
    degenerate weight vector). The max-Sharpe optimizer's internal
    objective function catches this and returns sentinel 1e10.

    Args:
        w:             (N,) weights
        mu:            (N,) expected returns
        Sigma:         (N,N) covariance matrix
        risk_free_rate: annualized decimal (Q6 convention: 0.02 = 2%/yr)

    Returns:
        Python float (may be negative per Q3).

    Raises:
        ZeroDivisionError: when σ_p < 1e-12.
    """
    if not np.isfinite(risk_free_rate):
        raise ValueError(f"risk_free_rate must be finite, got {risk_free_rate}")
    _assert_label_alignment(w, mu, Sigma)
    r_p = portfolio_return(w, mu)
    sigma_p = portfolio_volatility(w, Sigma)
    if sigma_p < _VOLATILITY_ZERO_TOL:
        raise ZeroDivisionError(
            f"portfolio_sharpe: σ_p = {sigma_p:.3e} < {_VOLATILITY_ZERO_TOL:.0e}; "
            "weight vector is degenerate (near-zero variance)"
        )
    return float((r_p - risk_free_rate) / sigma_p)