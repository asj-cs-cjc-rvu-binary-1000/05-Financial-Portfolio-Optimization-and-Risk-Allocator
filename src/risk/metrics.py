"""
src/risk/metrics.py
Phase 5 — Task 5.3: Risk metrics (drawdown, historical VaR / CVaR).

Implements the five risk functions specified in
docs/risk_analysis_specification.md, Sections 2 and 3:

    drawdown_series(returns)             -> pd.Series
    maximum_drawdown(returns)            -> float          (signed, <= 0)
    historical_var(returns, c=0.95)      -> float          (positive loss)
    historical_cvar(returns, c=0.95)     -> float          (positive loss)
    portfolio_historical_returns(w, returns) -> pd.Series

Sign convention summary (see spec Section 2.4):
- drawdown_series and maximum_drawdown return signed values (<= 0).
- historical_var and historical_cvar return positive loss magnitudes
  (negative if the tail is positive — unusual but mathematically valid).

All functions operate on DAILY return series. No annualization happens here.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "drawdown_series",
    "maximum_drawdown",
    "historical_var",
    "historical_cvar",
    "portfolio_historical_returns",
]

_DEFAULT_CONFIDENCE = 0.95


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_confidence(c: float, name: str) -> float:
    """Validate that c is a float in (0, 1)."""
    if not isinstance(c, (int, float)):
        raise TypeError(f"{name}: confidence_level must be a number, got {type(c).__name__}")
    c_float = float(c)
    if not np.isfinite(c_float):
        raise ValueError(f"{name}: confidence_level must be finite, got {c_float}")
    if not (0.0 < c_float < 1.0):
        raise ValueError(f"{name}: confidence_level must be in (0, 1), got {c_float}")
    return c_float


def _validate_returns(returns: Union[pd.Series, pd.DataFrame], name: str) -> None:
    """Validate that returns is finite and non-empty."""
    if isinstance(returns, (pd.Series, pd.DataFrame)):
        if returns.empty:
            raise ValueError(f"{name}: returns is empty")
        if returns.isna().any().any() if isinstance(returns, pd.DataFrame) else returns.isna().any():
            raise ValueError(f"{name}: returns contains NaN")
        values = returns.values if isinstance(returns, pd.DataFrame) else returns.values
    else:
        arr = np.asarray(returns, dtype=float)
        if arr.size == 0:
            raise ValueError(f"{name}: returns is empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: returns contains non-finite values")
        return
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: returns contains non-finite values")


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def drawdown_series(returns: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Compute the peak-to-trough drawdown series from cumulative simple returns.

    Math:
        C_t = prod_{s<=t}(1 + r_s)
        P_t = max_{s<=t} C_s
        D_t = (C_t - P_t) / P_t

    D_t <= 0 by construction. Same index as the input.

    Args:
        returns: Daily simple returns. If DataFrame, each column is an
                 independent asset's return series; use `portfolio_historical_returns`
                 first if you need a single-column portfolio return series.

    Returns:
        pd.Series (if returns is Series) or pd.DataFrame (if returns is DataFrame),
        same index and labels, values in (-inf, 0].

    Raises:
        ValueError: if returns is empty or contains NaN / inf.
    """
    _validate_returns(returns, "drawdown_series")

    if isinstance(returns, pd.DataFrame):
        cum = (1.0 + returns).cumprod()
        running_max = cum.cummax()
        return (cum - running_max) / running_max

    # pd.Series
    cum = (1.0 + returns).cumprod()
    running_max = cum.cummax()
    return (cum - running_max) / running_max


def maximum_drawdown(returns: Union[pd.Series, pd.DataFrame]) -> float:
    """
    Maximum drawdown: min of drawdown_series, a signed value <= 0.

    IDENTITY (test invariant):
        maximum_drawdown(returns) == drawdown_series(returns).min()

    Args:
        returns: Daily simple returns. If DataFrame, computes the worst
                 column drawdown (minimum across all columns and all dates).

    Returns:
        float (Python scalar), signed, <= 0. E.g., -0.25 for a 25% peak-to-trough decline.
    """
    dd = drawdown_series(returns)
    if isinstance(dd, pd.DataFrame):
        return float(dd.min().min())
    return float(dd.min())


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------


def historical_var(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = _DEFAULT_CONFIDENCE,
) -> float:
    """
    Historical Value-at-Risk at the given confidence level.

    Math:
        VaR_c = -Q_{1-c}(R)  (quantile in RETURN space, negated for loss convention)

    Args:
        returns: Daily simple returns (Series or 1-D array).
        confidence_level: c in (0, 1). Default 0.95.

    Returns:
        float, positive loss magnitude (negative if the tail is positive).
    """
    c = _validate_confidence(confidence_level, "historical_var")
    _validate_returns(returns, "historical_var")

    arr = returns.values if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float)
    quantile_alpha = 1.0 - c
    q = float(np.quantile(arr, quantile_alpha))
    return -q


def historical_cvar(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = _DEFAULT_CONFIDENCE,
) -> float:
    """
    Historical CVaR / Expected Shortfall: negative mean of returns at or below VaR threshold.

    Math:
        CVaR_c = -E[R | R <= Q_{1-c}(R)]

    Args:
        returns: Daily simple returns.
        confidence_level: c in (0, 1). Default 0.95.

    Returns:
        float, positive loss magnitude (negative if the tail is positive).
        CVaR >= VaR in magnitude (average bad day is worse than the threshold day).
    """
    c = _validate_confidence(confidence_level, "historical_cvar")
    _validate_returns(returns, "historical_cvar")

    arr = returns.values if isinstance(returns, pd.Series) else np.asarray(returns, dtype=float)
    quantile_alpha = 1.0 - c
    threshold = float(np.quantile(arr, quantile_alpha))

    tail = arr[arr <= threshold]
    if tail.size < 2:
        warnings.warn(
            f"historical_cvar: only {tail.size} observations at or below "
            f"the {quantile_alpha:.0%} quantile; returning VaR as CVaR",
            RuntimeWarning,
        )
        return -threshold
    return -float(tail.mean())


# ---------------------------------------------------------------------------
# Portfolio return series (used by Task 5.7)
# ---------------------------------------------------------------------------


def portfolio_historical_returns(
    w: Union[pd.Series, np.ndarray],
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Compute the daily portfolio return series for a fixed weight vector.

    Math:
        R_p,t = w^T r_t  for each day t

    Assumptions (spec Section 3):
        - Fixed target weights w (no re-optimization over history).
        - Daily rebalancing is implicit.
        - No transaction costs.
        - Not a historical trading backtest.

    Args:
        w: (N,) weight vector, aligned with returns.columns.
        returns: (T, N) DataFrame of daily returns, DatetimeIndex.

    Returns:
        pd.Series, DatetimeIndex, name = "portfolio_return".

    Raises:
        AssertionError: if w.index (when Series) does not match returns.columns.
        ValueError: if shape mismatch.
    """
    if isinstance(w, pd.Series):
        if not w.index.equals(returns.columns):
            raise AssertionError(
                f"Label mismatch: w.index={list(w.index)!r}, "
                f"returns.columns={list(returns.columns)!r}"
            )
        w_arr = w.to_numpy(dtype=float, copy=False)
    else:
        w_arr = np.asarray(w, dtype=float)

    if w_arr.ndim != 1:
        raise ValueError(f"portfolio_historical_returns: w must be 1-D, got {w_arr.shape}")
    if w_arr.shape[0] != returns.shape[1]:
        raise ValueError(
            f"portfolio_historical_returns: w has {w_arr.shape[0]} entries, "
            f"returns has {returns.shape[1]} columns"
        )

    R_p = returns.to_numpy(dtype=float) @ w_arr
    return pd.Series(R_p, index=returns.index, name="portfolio_return")