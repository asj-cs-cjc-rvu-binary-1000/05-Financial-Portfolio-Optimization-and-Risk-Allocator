"""
tests/test_optimization_pipeline.py
Phase 4 — Task 4.9: Offline tests for the comparison pipeline.

Verifies:
- Shape and row/column ordering invariants.
- Integration: the EqualWeight row matches equal_weight_portfolio exactly.
- Strategy ordering: MinVolatility.volatility <= EqualWeight.volatility;
  MaxSharpe.sharpe >= EqualWeight.sharpe (spec Section 2.1, 2.2).
- Target return row added only when target_return is given.
- Failure rows appear with success=False and nan numeric cells.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.optimization_pipeline import compare_portfolios
from src.optimization.benchmarks import equal_weight_portfolio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_asset():
    labels = ["A", "B", "C"]
    mu = pd.Series([0.10, 0.15, 0.20], index=labels)
    Sigma = pd.DataFrame(
        [[0.040, 0.010, 0.005],
         [0.010, 0.090, 0.015],
         [0.005, 0.015, 0.160]],
        index=labels, columns=labels,
    )
    return {"labels": labels, "mu": mu, "Sigma": Sigma}


METRIC_COLS = ["return", "volatility", "variance", "sharpe", "success", "n_iter"]
ROW_ORDER_3 = ["EqualWeight", "MinVolatility", "MaxSharpe"]
ROW_ORDER_4 = ROW_ORDER_3 + ["TargetReturn"]


# ---------------------------------------------------------------------------
# Shape and ordering
# ---------------------------------------------------------------------------


def test_three_rows_without_target(three_asset):
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    assert list(df.index) == ROW_ORDER_3


def test_four_rows_with_target(three_asset):
    df = compare_portfolios(
        three_asset["mu"], three_asset["Sigma"], 0.02, target_return=0.15
    )
    assert list(df.index) == ROW_ORDER_4


def test_column_order_is_metrics_then_tickers(three_asset):
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    expected = METRIC_COLS + list(three_asset["labels"])
    assert list(df.columns) == expected


def test_ticker_order_matches_mu_index(three_asset):
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    assert list(df.columns[6:]) == list(three_asset["mu"].index)


def test_dtypes_are_stable(three_asset):
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    for c in ["return", "volatility", "variance", "sharpe"]:
        assert df[c].dtype == np.float64, f"{c} dtype {df[c].dtype}"
    assert df["success"].dtype == bool
    assert df["n_iter"].dtype == np.int64
    for t in three_asset["labels"]:
        assert df[t].dtype == np.float64


# ---------------------------------------------------------------------------
# Integration with the benchmark (single source of truth across phases)
# ---------------------------------------------------------------------------


def test_equal_weight_row_matches_benchmark_function(three_asset):
    """
    Design invariant: the EqualWeight row must exactly equal the output
    of equal_weight_portfolio. If they ever drift, we have two sources
    of truth for the same portfolio — a Phase 6 dashboard bug waiting
    to happen.
    """
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    eq = equal_weight_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)

    assert df.loc["EqualWeight", "return"]     == pytest.approx(eq["return"], abs=1e-12)
    assert df.loc["EqualWeight", "volatility"] == pytest.approx(eq["volatility"], abs=1e-12)
    assert df.loc["EqualWeight", "variance"]   == pytest.approx(eq["variance"], abs=1e-12)
    assert df.loc["EqualWeight", "sharpe"]     == pytest.approx(eq["sharpe"], abs=1e-12)
    for t in three_asset["labels"]:
        assert df.loc["EqualWeight", t] == pytest.approx(eq["weights"][t], abs=1e-12)


# ---------------------------------------------------------------------------
# Mathematical invariants
# ---------------------------------------------------------------------------


def test_min_vol_volatility_no_worse_than_equal_weight(three_asset):
    """Min-var is by definition the global min volatility under the constraints."""
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    assert df.loc["MinVolatility", "volatility"] <= df.loc["EqualWeight", "volatility"] + 1e-8


def test_max_sharpe_no_worse_than_equal_weight(three_asset):
    """Max-Sharpe is by definition the global max Sharpe under the constraints."""
    df = compare_portfolios(three_asset["mu"], three_asset["Sigma"], 0.02)
    assert df.loc["MaxSharpe", "sharpe"] >= df.loc["EqualWeight", "sharpe"] - 1e-8


def test_all_successful_rows_sum_to_one(three_asset):
    df = compare_portfolios(
        three_asset["mu"], three_asset["Sigma"], 0.02, target_return=0.15
    )
    tickers = list(three_asset["labels"])
    for idx in df.index:
        if df.loc[idx, "success"]:
            assert df.loc[idx, tickers].sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Failure rows still appear (with nan cells)
# ---------------------------------------------------------------------------


def test_target_out_of_range_still_appears_with_nan(three_asset):
    """
    A target outside [mu.min(), mu.max()] is a spec-defined failure
    (Q1/Q2). The pipeline must still include the TargetReturn row — its
    numeric columns are nan and success=False — so downstream code sees
    an explicit "this row failed" rather than a silently missing row.
    """
    df = compare_portfolios(
        three_asset["mu"], three_asset["Sigma"], 0.02, target_return=0.999
    )
    assert "TargetReturn" in df.index
    assert df.loc["TargetReturn", "success"] == False  # noqa: E712
    assert np.isnan(df.loc["TargetReturn", "return"])
    assert np.isnan(df.loc["TargetReturn", "volatility"])
    assert np.isnan(df.loc["TargetReturn", "variance"])


# ---------------------------------------------------------------------------
# Input validation passthrough
# ---------------------------------------------------------------------------


def test_non_finite_mu_raises(three_asset):
    mu = three_asset["mu"].copy()
    mu.iloc[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        compare_portfolios(mu, three_asset["Sigma"], 0.02)


def test_label_mismatch_raises():
    mu = pd.Series([0.10, 0.15], index=["A", "B"])
    Sigma = pd.DataFrame(np.eye(2) * 0.05, index=["A", "C"], columns=["A", "C"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        compare_portfolios(mu, Sigma, 0.02)


# ---------------------------------------------------------------------------
# Determinism (spec Section 8)
# ---------------------------------------------------------------------------


def test_two_calls_are_byte_identical(three_asset):
    df1 = compare_portfolios(
        three_asset["mu"], three_asset["Sigma"], 0.02, target_return=0.15
    )
    df2 = compare_portfolios(
        three_asset["mu"], three_asset["Sigma"], 0.02, target_return=0.15
    )
    pd.testing.assert_frame_equal(df1, df2, check_exact=True)