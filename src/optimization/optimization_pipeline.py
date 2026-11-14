"""
src/optimization/optimization_pipeline.py
Phase 4 — Task 4.9: Portfolio comparison pipeline.

A single function that runs every Phase 4 portfolio (equal-weight benchmark,
minimum-variance, maximum-Sharpe, and optionally a target-return portfolio)
and returns one tidy pd.DataFrame. The output is the canonical comparison
table that Phase 5's frontier plot, Phase 6's dashboard, and Phase 7's
notebook all consume.

Specification contract (see docs/optimization_specification.md):
- Section 7 (Return-Value Contract): every optimizer and the benchmark
  return the same dict. This function reads only the standard keys
  (weights, return, volatility, variance, sharpe, success, n_iter).
- Section 8 (Determinism): given the same (mu, Sigma, r_f, target), the
  table is byte-identical across runs on the same environment.

Row order is FIXED (do not sort): EqualWeight, MinVolatility, MaxSharpe,
[TargetReturn]. Ticker column order follows mu.index exactly.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from src.optimization.benchmarks import equal_weight_portfolio
from src.optimization.optimizer import (
    maximum_sharpe_portfolio,
    minimum_variance_portfolio,
    target_return_portfolio,
)

__all__ = ["compare_portfolios"]

# Fixed row order — do not reorder.
_ROW_ORDER_BASE = ["EqualWeight", "MinVolatility", "MaxSharpe"]
_ROW_ORDER_WITH_TARGET = _ROW_ORDER_BASE + ["TargetReturn"]

# Fixed metric column order — do not reorder.
_METRIC_COLS = ["return", "volatility", "variance", "sharpe", "success", "n_iter"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_labels(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
) -> pd.Index:
    """Same label rule as benchmarks.py — prefer mu.index, then Sigma.index."""
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
        "compare_portfolios: at least one of mu / Sigma must carry ticker "
        "labels (pass as pd.Series / pd.DataFrame)"
    )


def _result_to_row(result: dict, labels: pd.Index) -> dict:
    """
    Flatten a standard Phase 4 result dict into a flat row keyed by the
    metric columns plus one column per ticker (the ticker columns hold
    the weight for that ticker).

    Handles weights=None (spec 5.4 pre-solver failure) by writing nan
    for all ticker weight cells.
    """
    row: dict = {
        "return": result["return"],
        "volatility": result["volatility"],
        "variance": result["variance"],
        "sharpe": (result["sharpe"] if result["sharpe"] is not None else np.nan),
        "success": bool(result["success"]),
        "n_iter": int(result["n_iter"]),
    }

    w = result["weights"]
    if w is None:
        for t in labels:
            row[t] = np.nan
    else:
        # Match labels strictly — do not silently reindex.
        for t in labels:
            row[t] = float(w.loc[t])

    return row


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_portfolios(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    risk_free_rate: float,
    target_return: Optional[float] = None,
) -> pd.DataFrame:
    """
    Run every Phase 4 portfolio strategy and return a tidy comparison table.

    Args:
        mu:             (N,) expected returns (annualized)
        Sigma:          (N,N) covariance matrix (annualized)
        risk_free_rate: annualized decimal (0.02 = 2%/yr). Required —
                        this function always computes Sharpe.
        target_return:  optional scalar target for the target-return
                        optimizer. If provided, a fourth row is added.
                        If None, the target row is omitted.

    Returns:
        pd.DataFrame with:
            index   — strategy names, in the fixed order
                      ["EqualWeight", "MinVolatility", "MaxSharpe", "TargetReturn"]
                      (TargetReturn only if target_return is not None)
            columns — ["return", "volatility", "variance", "sharpe",
                       "success", "n_iter"] followed by one column per
                      ticker in mu.index order.
        Dtypes: numeric columns are float64; "success" is bool; "n_iter"
        is int64; ticker columns are float64.

    Raises:
        ValueError: on non-finite inputs or missing labels.
        AssertionError: on label mismatch between mu and Sigma.
    """
    labels = _extract_labels(mu, Sigma)
    ticker_cols = list(labels)

    rows: dict[str, dict] = {}

    # Row 1 — equal-weight benchmark (always successful if labels exist).
    eq = equal_weight_portfolio(mu, Sigma, risk_free_rate)
    rows["EqualWeight"] = _result_to_row(eq, labels)

    # Row 2 — minimum variance.
    mv = minimum_variance_portfolio(mu, Sigma, risk_free_rate)
    rows["MinVolatility"] = _result_to_row(mv, labels)

    # Row 3 — maximum Sharpe (risk_free_rate required — always provided here).
    ms = maximum_sharpe_portfolio(mu, Sigma, risk_free_rate)
    rows["MaxSharpe"] = _result_to_row(ms, labels)

    # Row 4 (optional) — target-return portfolio.
    if target_return is not None:
        tr = target_return_portfolio(
            mu, Sigma, target_return=float(target_return),
            risk_free_rate=risk_free_rate,
        )
        rows["TargetReturn"] = _result_to_row(tr, labels)
        row_order = _ROW_ORDER_WITH_TARGET
    else:
        row_order = _ROW_ORDER_BASE

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(index=row_order, columns=_METRIC_COLS + ticker_cols)

    # Enforce dtypes explicitly (protects against numpy scalar drift).
    for c in ("return", "volatility", "variance", "sharpe", *ticker_cols):
        df[c] = df[c].astype(float)
    df["success"] = df["success"].astype(bool)
    df["n_iter"] = df["n_iter"].astype(np.int64)

    return df

# ---------------------------------------------------------------------------
# CLI entry point — python -m src.optimization.optimization_pipeline
# ---------------------------------------------------------------------------


def _main() -> int:
    """
    Load the latest cleaned prices, run the returns pipeline, compute the
    covariance matrix, run all four portfolios, print and persist the
    comparison table to docs/optimization_summary.md.

    Returns 0 on success, non-zero on failure.
    """
    import sys
    from pathlib import Path

    import tabulate

    from src.data.stock_data import load_prices
    from src.linear_algebra.covariance import compute_covariance_matrix
    from src.preprocessing.returns import process_returns_pipeline

    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "data" / "processed"
    docs_dir = repo_root / "docs"

    if not data_dir.exists():
        print(f"[error] {data_dir} not found", file=sys.stderr)
        return 1

    csvs = sorted(data_dir.glob("prices_clean_*.csv"))
    if not csvs:
        print(
            f"[error] no processed prices CSV in {data_dir}; run "
            f"`python -m src.data.stock_data` first",
            file=sys.stderr,
        )
        return 1
    latest = csvs[-1]

    print(f"[info] loading {latest.name}")
    prices = load_prices(str(latest))

    print("[info] computing returns and expected returns...")
    returns, mu, sigma, _ = process_returns_pipeline(prices)

    print("[info] computing annualized covariance matrix...")
    Sigma = compute_covariance_matrix(returns, annualize=True)

    R_F = 0.02
    print(f"[info] running comparison at r_f = {R_F} ...")
    df = compare_portfolios(mu, Sigma, risk_free_rate=R_F)
    df_rounded = df.round(4)

    print("\n=== Portfolio Comparison ===")
    print(df_rounded.to_string())

    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "optimization_summary.md"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 4 — Portfolio Optimization Summary\n\n")
        fh.write("**Live run** — `python -m src.optimization.optimization_pipeline`\n\n")
        fh.write(f"- Data file: `data/processed/{latest.name}`\n")
        fh.write(f"- Assets: {', '.join(map(str, mu.index))}\n")
        fh.write(f"- Risk-free rate: {R_F} (annualized decimal, 2%/yr)\n\n")
        fh.write("## Comparison Table\n\n")
        fh.write(tabulate.tabulate(df_rounded, headers="keys", tablefmt="github"))
        fh.write("\n\n## Interpretation\n\n")
        _write_interpretation(fh, df)

    print(f"[info] wrote {out}")
    return 0


def _write_interpretation(fh, df: pd.DataFrame) -> None:
    """
    Write a short plain-English interpretation block. Claims are read
    from the actual DataFrame — no narrative that is not backed by a
    table cell.
    """
    # Best Sharpe row
    best_sharpe_row = df["sharpe"].astype(float).idxmax()
    best_sharpe_val = float(df.loc[best_sharpe_row, "sharpe"])

    # Min-volatility row
    low_vol_row = df["volatility"].astype(float).idxmin()
    low_vol_val = float(df.loc[low_vol_row, "volatility"])

    # Concentration: how many non-dust weights (> 0.05) in MaxSharpe row
    ticker_cols = [c for c in df.columns
                   if c not in ("return", "volatility", "variance",
                                "sharpe", "success", "n_iter")]
    if "MaxSharpe" in df.index:
        ms_w = df.loc["MaxSharpe", ticker_cols].astype(float)
        ms_active = int((ms_w > 0.05).sum())
    else:
        ms_active = -1

    fh.write(
        f"- **Highest Sharpe ratio:** `{best_sharpe_row}` "
        f"with Sharpe = {best_sharpe_val:.4f}.\n"
    )
    fh.write(
        f"- **Lowest volatility:** `{low_vol_row}` "
        f"with σ_p = {low_vol_val:.4f}.\n"
    )
    if ms_active >= 0:
        fh.write(
            f"- **Max-Sharpe concentration:** "
            f"{ms_active} of {len(ticker_cols)} assets hold weight > 5% "
            f"(max-Sharpe typically concentrates; equal-weight never does).\n"
        )
    fh.write(
        "- **Interpretation:** the min-variance row has the smallest "
        "volatility by construction; the max-Sharpe row has the largest "
        "Sharpe by construction. The equal-weight row is the naive "
        "benchmark — if an optimizer does not beat it on its own metric, "
        "investigate rather than assume a bug (spec Section 8: SLSQP is a "
        "local solver and the max-Sharpe problem is non-convex).\n"
    )


if __name__ == "__main__":
    raise SystemExit(_main())