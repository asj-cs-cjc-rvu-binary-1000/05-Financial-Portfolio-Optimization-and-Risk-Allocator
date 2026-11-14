"""
src/risk/frontier.py
Phase 5 — Task 5.4: Efficient frontier generator.

Sweeps target returns across [mu_min, mu_max] using Phase 4's
`target_return_portfolio`, splits the sweep at mu_min_var into the
efficient (upper) branch and the inefficient (lower) branch, and
returns both as tidy DataFrames plus metadata.

See docs/risk_analysis_specification.md Section 4 for the full contract.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from src.optimization.benchmarks import equal_weight_portfolio
from src.optimization.optimizer import (
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    target_return_portfolio,
)

__all__ = ["efficient_frontier", "analyze_frontier"]

# Schema columns shared with Phase 4 comparison pipeline.
_METRIC_COLS = ["return", "volatility", "variance", "success", "n_iter"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_labels(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
) -> pd.Index:
    """Same label rule as Phase 4 benchmarks."""
    mu_idx = mu.index if isinstance(mu, pd.Series) else None
    sigma_idx = Sigma.index if isinstance(Sigma, pd.DataFrame) else None

    if mu_idx is not None and sigma_idx is not None and not mu_idx.equals(sigma_idx):
        raise AssertionError(
            f"Label mismatch: mu.index={list(mu_idx)!r}, "
            f"Sigma.index={list(sigma_idx)!r}"
        )
    if mu_idx is not None:
        return mu_idx
    if sigma_idx is not None:
        return sigma_idx
    raise ValueError(
        "efficient_frontier: at least one of mu / Sigma must carry ticker labels"
    )


def _sweep(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    targets: np.ndarray,
    tickers: list[str],
) -> pd.DataFrame:
    """Run target_return_portfolio for each target. Return a tidy DataFrame."""
    rows = []
    for t in targets:
        r = target_return_portfolio(mu, Sigma, target_return=float(t))
        row = {
            "target_return": float(t),
            "return": float(r["return"]),
            "volatility": float(r["volatility"]),
            "variance": float(r["variance"]),
            "success": bool(r["success"]),
            "n_iter": int(r["n_iter"]),
        }
        w = r["weights"]
        if w is None:
            for ticker in tickers:
                row[ticker] = float("nan")
        else:
            for ticker in tickers:
                row[ticker] = float(w.loc[ticker])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["target_return", *_METRIC_COLS, *tickers]]
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def efficient_frontier(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    n_points: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Generate the Markowitz efficient frontier in two branches.

    Args:
        mu:       (N,) annualized expected returns, Series or ndarray.
        Sigma:    (N,N) annualized covariance matrix, DataFrame or ndarray.
        n_points: number of points per branch (default 50).

    Returns:
        (efficient_df, inefficient_df, metadata):
            efficient_df    — n_points targets in [mu_min_var, mu_max]
            inefficient_df  — n_points targets in [mu_min, mu_min_var]
            metadata        — dict with boundaries and success counts
    """
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError(f"efficient_frontier: n_points must be int >= 2, got {n_points!r}")

    labels = _extract_labels(mu, Sigma)
    tickers = list(labels)

    mu_arr = mu.to_numpy() if isinstance(mu, pd.Series) else np.asarray(mu, dtype=float)
    mu_min = float(mu_arr.min())
    mu_max = float(mu_arr.max())

    # Anchor: minimum-variance portfolio (Task 4.6)
    mv = minimum_variance_portfolio(mu, Sigma)
    if not mv["success"]:
        raise RuntimeError(
            f"efficient_frontier: minimum_variance_portfolio failed "
            f"({mv['message']}); cannot anchor the frontier"
        )
    mu_min_var = float(mv["return"])

    # Sanity: mu_min_var should be in [mu_min, mu_max]
    if not (mu_min - 1e-9 <= mu_min_var <= mu_max + 1e-9):
        raise RuntimeError(
            f"efficient_frontier: mu_min_var = {mu_min_var:.6f} lies outside "
            f"[mu_min, mu_max] = [{mu_min:.6f}, {mu_max:.6f}]; "
            "the min-variance portfolio returned an infeasible expected return"
        )

    # Efficient branch: [mu_min_var, mu_max]
    eff_targets = np.linspace(mu_min_var, mu_max, n_points)
    efficient_df = _sweep(mu, Sigma, eff_targets, tickers)

    # Inefficient branch: [mu_min, mu_min_var]
    ineff_targets = np.linspace(mu_min, mu_min_var, n_points)
    inefficient_df = _sweep(mu, Sigma, ineff_targets, tickers)

    metadata = {
        "mu_min": mu_min,
        "mu_min_var": mu_min_var,
        "mu_max": mu_max,
        "n_points": int(n_points),
        "n_success_efficient": int(efficient_df["success"].sum()),
        "n_success_inefficient": int(inefficient_df["success"].sum()),
        "n_failed_efficient": int((~efficient_df["success"]).sum()),
        "n_failed_inefficient": int((~inefficient_df["success"]).sum()),
    }
    return efficient_df, inefficient_df, metadata

# ---------------------------------------------------------------------------
# Task 5.5 — Frontier analysis
# ---------------------------------------------------------------------------


def analyze_frontier(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: float,
    n_points: int = 50,
) -> dict:
    """
    Run the frontier generation and identify the key points for plotting.

    Returns a dict with:
        efficient_df       — output of efficient_frontier[0]
        inefficient_df     — output of efficient_frontier[1]
        metadata           — output of efficient_frontier[2]
        min_var            — dict: return, volatility, weights (from minimum_variance_portfolio)
        max_sharpe         — dict: return, volatility, sharpe, weights (from maximum_sharpe_portfolio)
        equal_weight       — dict: return, volatility, sharpe, weights (from equal_weight_portfolio)
        assets             — pd.DataFrame with columns [ticker, mu, sigma],
                              sorted by ticker (sigma from √diag(Σ))

    Args:
        mu:             (N,) annualized expected returns, Series or ndarray.
        Sigma:          (N,N) annualized covariance matrix, DataFrame or ndarray.
        risk_free_rate: annualized decimal.
        n_points:       number of points per frontier branch (default 50).

    Returns:
        dict — see above.
    """
    if not np.isfinite(risk_free_rate):
        raise ValueError(f"risk_free_rate must be finite, got {risk_free_rate}")

    labels = _extract_labels(mu, Sigma)

    # 1. Efficient frontier
    efficient_df, inefficient_df, metadata = efficient_frontier(mu, Sigma, n_points=n_points)

    # 2. Min-variance, max-Sharpe, equal-weight anchor portfolios
    mv = minimum_variance_portfolio(mu, Sigma, risk_free_rate)
    if not mv["success"]:
        raise RuntimeError(f"analyze_frontier: minimum_variance_portfolio failed: {mv['message']}")
    min_var = {
        "return": float(mv["return"]),
        "volatility": float(mv["volatility"]),
        "weights": mv["weights"],
    }

    ms = maximum_sharpe_portfolio(mu, Sigma, risk_free_rate)
    if not ms["success"]:
        raise RuntimeError(f"analyze_frontier: maximum_sharpe_portfolio failed: {ms['message']}")
    max_sharpe = {
        "return": float(ms["return"]),
        "volatility": float(ms["volatility"]),
        "sharpe": float(ms["sharpe"]),
        "weights": ms["weights"],
    }

    eq = equal_weight_portfolio(mu, Sigma, risk_free_rate)
    equal_weight = {
        "return": float(eq["return"]),
        "volatility": float(eq["volatility"]),
        "sharpe": float(eq["sharpe"]),
        "weights": eq["weights"],
    }

    # 3. Individual asset data (for scatter)
    mu_arr = mu.to_numpy(dtype=float) if isinstance(mu, pd.Series) else np.asarray(mu, dtype=float)
    if isinstance(Sigma, pd.DataFrame):
        Sigma_arr = Sigma.to_numpy(dtype=float)
    else:
        Sigma_arr = np.asarray(Sigma, dtype=float)
    sigma_arr = np.sqrt(np.diag(Sigma_arr))

    assets = pd.DataFrame({
        "ticker": list(labels),
        "mu": mu_arr,
        "sigma": sigma_arr,
    })

    return {
        "efficient_df": efficient_df,
        "inefficient_df": inefficient_df,
        "metadata": metadata,
        "min_var": min_var,
        "max_sharpe": max_sharpe,
        "equal_weight": equal_weight,
        "assets": assets,
    }