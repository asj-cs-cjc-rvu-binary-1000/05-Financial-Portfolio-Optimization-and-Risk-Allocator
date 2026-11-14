"""
src/risk/comparison.py
Phase 5 — Task 5.7: Risk-metric comparison across portfolios.

A sibling of `src.optimization.optimization_pipeline.compare_portfolios` that
adds the Phase 5 risk columns (VaR, CVaR, MDD) to the standard Phase 4 metrics.

Assumptions (spec Section 3):
    Portfolio return series = w^T r_t for each historical day, with w fixed.
    This is a static allocation analysis, not a backtest.
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
from src.risk.metrics import (
    historical_cvar,
    historical_var,
    maximum_drawdown,
    portfolio_historical_returns,
)

__all__ = ["compare_risk_metrics"]

_ROW_ORDER_BASE = ["EqualWeight", "MinVolatility", "MaxSharpe"]
_ROW_ORDER_WITH_TARGET = _ROW_ORDER_BASE + ["TargetReturn"]

_METRIC_COLS = [
    "return", "volatility", "sharpe",
    "var_95", "cvar_95", "max_drawdown",
    "success",
]


def _extract_labels(mu, Sigma) -> pd.Index:
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
    raise ValueError("compare_risk_metrics: at least one of mu / Sigma must carry labels")


def _risk_row(
    result: dict,
    returns: pd.DataFrame,
    labels: pd.Index,
    confidence: float,
) -> dict:
    """Build one row: Phase 4 metrics + Phase 5 risk metrics + weights."""
    row: dict = {
        "return": float(result["return"]),
        "volatility": float(result["volatility"]),
        "sharpe": float(result["sharpe"]) if result["sharpe"] is not None else float("nan"),
        "var_95": float("nan"),
        "cvar_95": float("nan"),
        "max_drawdown": float("nan"),
        "success": bool(result["success"]),
    }

    w = result["weights"]
    if w is None or not result["success"]:
        for t in labels:
            row[t] = float("nan")
        return row

    R_p = portfolio_historical_returns(w, returns)
    row["var_95"] = float(historical_var(R_p, confidence))
    row["cvar_95"] = float(historical_cvar(R_p, confidence))
    row["max_drawdown"] = float(maximum_drawdown(R_p))

    for t in labels:
        row[t] = float(w.loc[t])
    return row


def compare_risk_metrics(
    mu: Union[pd.Series, np.ndarray],
    Sigma: Union[pd.DataFrame, np.ndarray],
    returns: pd.DataFrame,
    risk_free_rate: float,
    target_return: Optional[float] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    Compute return, volatility, Sharpe, VaR, CVaR, and MDD for each portfolio.

    Args:
        mu:             (N,) annualized expected returns
        Sigma:          (N,N) annualized covariance matrix
        returns:        (T,N) daily simple returns DataFrame
        risk_free_rate: annualized decimal (0.02 = 2%/yr)
        target_return:  optional target-return portfolio; if provided, an
                        extra row is added
        confidence_level: VaR/CVaR confidence (default 0.95)

    Returns:
        pd.DataFrame with rows (fixed order)
            [EqualWeight, MinVolatility, MaxSharpe, (TargetReturn)]
        and columns
            [return, volatility, sharpe, var_95, cvar_95, max_drawdown,
             success] + one per ticker.

        - All numeric columns are float64 (including var_95 / cvar_95 /
          max_drawdown), nan when the row failed.
        - success is bool.
        - Sign conventions: var_95 and cvar_95 are positive loss magnitudes;
          max_drawdown is a signed value ≤ 0 (see spec Section 2.4).

    Raises:
        ValueError: on non-finite inputs or invalid confidence.
        AssertionError: on label mismatch between mu and Sigma, or between
                        mu/Sigma and returns.columns.
    """
    if not np.isfinite(risk_free_rate):
        raise ValueError(f"compare_risk_metrics: risk_free_rate must be finite, got {risk_free_rate}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"compare_risk_metrics: confidence_level must be in (0, 1), got {confidence_level}")

    labels = _extract_labels(mu, Sigma)
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("compare_risk_metrics: returns must be a pd.DataFrame")
    if not returns.columns.equals(labels):
        raise AssertionError(
            f"Label mismatch: returns.columns={list(returns.columns)!r}, "
            f"labels={list(labels)!r}"
        )

    tickers = list(labels)

    rows: dict[str, dict] = {}

    eq = equal_weight_portfolio(mu, Sigma, risk_free_rate)
    rows["EqualWeight"] = _risk_row(eq, returns, labels, confidence_level)

    mv = minimum_variance_portfolio(mu, Sigma, risk_free_rate)
    rows["MinVolatility"] = _risk_row(mv, returns, labels, confidence_level)

    ms = maximum_sharpe_portfolio(mu, Sigma, risk_free_rate)
    rows["MaxSharpe"] = _risk_row(ms, returns, labels, confidence_level)

    if target_return is not None:
        tr = target_return_portfolio(
            mu, Sigma, target_return=float(target_return),
            risk_free_rate=risk_free_rate,
        )
        rows["TargetReturn"] = _risk_row(tr, returns, labels, confidence_level)
        row_order = _ROW_ORDER_WITH_TARGET
    else:
        row_order = _ROW_ORDER_BASE

    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(index=row_order, columns=_METRIC_COLS + tickers)

    for c in ("return", "volatility", "sharpe", "var_95", "cvar_95",
              "max_drawdown", *tickers):
        df[c] = df[c].astype(float)
    df["success"] = df["success"].astype(bool)

    return df

# ---------------------------------------------------------------------------
# CLI — python -m src.risk.comparison
# ---------------------------------------------------------------------------


def _write_interpretation(fh, df: pd.DataFrame, confidence: float) -> None:
    """
    Write a plain-English interpretation paragraph. Every claim is read
    from the DataFrame — no narrative that is not backed by a table cell.
    """
    idx_min_vol = df["volatility"].idxmin()
    idx_max_sharpe = df["sharpe"].idxmax()
    idx_min_var = df["var_95"].idxmin()
    idx_min_mdd = df["max_drawdown"].idxmax()  # least negative = best

    fh.write("## Interpretation\n\n")
    fh.write(
        f"- **Lowest volatility:** `{idx_min_vol}` "
        f"(σ_p = {df.loc[idx_min_vol, 'volatility']:.4f}).\n"
    )
    fh.write(
        f"- **Highest Sharpe:** `{idx_max_sharpe}` "
        f"(S_p = {df.loc[idx_max_sharpe, 'sharpe']:.4f}).\n"
    )
    fh.write(
        f"- **Lowest VaR{int(confidence*100)}:** `{idx_min_var}` "
        f"(VaR = {df.loc[idx_min_var, 'var_95']*100:.2f}%, "
        f"CVaR = {df.loc[idx_min_var, 'cvar_95']*100:.2f}%).\n"
    )
    fh.write(
        f"- **Shallowest max drawdown:** `{idx_min_mdd}` "
        f"(MDD = {df.loc[idx_min_mdd, 'max_drawdown']*100:.2f}%).\n\n"
    )
    fh.write(
        "**Reading the table:** VaR and CVaR are reported as positive loss "
        "magnitudes (industry convention). MDD is a signed peak-to-trough "
        "cumulative return (≤ 0), consistent with `drawdown_series`. A "
        "portfolio with the lowest volatility does not necessarily have "
        "the lowest VaR — VaR depends on the tail shape, not just the "
        "second moment. If MinVolatility does not have the smallest VaR on "
        "this data, that is a legitimate empirical result, not a bug.\n"
    )


def _main() -> int:
    """Load latest processed CSV, compute risk metrics, write docs/risk_analysis.md."""
    import sys
    from pathlib import Path

    import tabulate

    from src.data.stock_data import load_prices
    from src.linear_algebra.covariance import compute_covariance_matrix
    from src.preprocessing.returns import process_returns_pipeline

    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "data" / "processed"
    docs_dir = repo_root / "docs"

    csvs = sorted(data_dir.glob("prices_clean_*.csv"))
    if not csvs:
        print(f"[error] no processed prices CSV in {data_dir}; "
              f"run `python -m src.data.stock_data` first", file=sys.stderr)
        return 1
    latest = csvs[-1]

    print(f"[info] loading {latest.name}")
    prices = load_prices(str(latest))
    print("[info] computing returns and annualized covariance...")
    returns, mu, sigma, _ = process_returns_pipeline(prices)
    Sigma = compute_covariance_matrix(returns, annualize=True)

    R_F = 0.02
    CONF = 0.95
    print(f"[info] computing risk metrics at r_f = {R_F}, c = {CONF}...")
    df = compare_risk_metrics(mu, Sigma, returns, R_F, confidence_level=CONF)
    df_rounded = df.round(4)

    print("\n=== Risk Comparison ===")
    print(df_rounded.to_string())

    docs_dir.mkdir(parents=True, exist_ok=True)
    out = docs_dir / "risk_analysis.md"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# Phase 5 — Risk Analysis Summary\n\n")
        fh.write("**Live run** — `python -m src.risk.comparison`\n\n")
        fh.write(f"- Data file: `data/processed/{latest.name}`\n")
        fh.write(f"- Assets: {', '.join(map(str, mu.index))}\n")
        fh.write(f"- Risk-free rate: {R_F} (annualized decimal, 2%/yr)\n")
        fh.write(f"- Confidence level: {CONF:.0%}\n")
        fh.write(f"- Horizon: daily (no √252 scaling)\n\n")
        fh.write("## Comparison Table\n\n")
        fh.write(tabulate.tabulate(df_rounded, headers="keys", tablefmt="github"))
        fh.write("\n\n")
        _write_interpretation(fh, df, CONF)

    print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())