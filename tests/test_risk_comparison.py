"""
tests/test_risk_comparison.py
Phase 5 — Task 5.7: offline tests for compare_risk_metrics.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.comparison import compare_risk_metrics


@pytest.fixture
def small_universe():
    labels = ["A", "B", "C"]
    mu = pd.Series([0.10, 0.15, 0.20], index=labels)
    Sigma = pd.DataFrame(
        [[0.040, 0.010, 0.005],
         [0.010, 0.090, 0.015],
         [0.005, 0.015, 0.160]],
        index=labels, columns=labels,
    )
    rng = np.random.default_rng(seed=7)
    daily = rng.normal(0.0004, 0.01, size=(300, 3))
    returns = pd.DataFrame(daily, columns=labels,
                           index=pd.date_range("2020-01-01", periods=300))
    return {"labels": labels, "mu": mu, "Sigma": Sigma, "returns": returns}


def test_three_rows_without_target(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    assert list(df.index) == ["EqualWeight", "MinVolatility", "MaxSharpe"]


def test_four_rows_with_target(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02, target_return=0.15,
    )
    assert list(df.index) == ["EqualWeight", "MinVolatility", "MaxSharpe", "TargetReturn"]


def test_columns(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    expected = ["return", "volatility", "sharpe",
                "var_95", "cvar_95", "max_drawdown", "success",
                *small_universe["labels"]]
    assert list(df.columns) == expected


def test_var_positive_loss(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    # VaR should be ≥ 0 on any healthy portfolio (unless the 5% quantile is positive)
    assert (df["var_95"].dropna() >= 0).all()


def test_cvar_greater_equal_var(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    for idx in df.index:
        assert df.loc[idx, "cvar_95"] >= df.loc[idx, "var_95"] - 1e-12


def test_max_drawdown_nonpositive(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    assert (df["max_drawdown"].dropna() <= 0).all()


def test_min_vol_has_lowest_vol(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    assert df.loc["MinVolatility", "volatility"] <= df.loc["EqualWeight", "volatility"] + 1e-8


def test_max_sharpe_has_highest_sharpe(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02,
    )
    assert df.loc["MaxSharpe", "sharpe"] >= df.loc["EqualWeight", "sharpe"] - 1e-8


def test_weights_sum_to_one_when_success(small_universe):
    df = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02, target_return=0.15,
    )
    tickers = small_universe["labels"]
    for idx in df.index:
        if df.loc[idx, "success"]:
            assert df.loc[idx, tickers].sum() == pytest.approx(1.0, abs=1e-6)


def test_label_mismatch_returns_vs_mu_raises(small_universe):
    returns_bad = small_universe["returns"].rename(columns={"A": "X"})
    with pytest.raises(AssertionError, match="Label mismatch"):
        compare_risk_metrics(
            small_universe["mu"], small_universe["Sigma"],
            returns_bad, 0.02,
        )


def test_invalid_confidence_raises(small_universe):
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        compare_risk_metrics(
            small_universe["mu"], small_universe["Sigma"],
            small_universe["returns"], 0.02, confidence_level=1.2,
        )


def test_deterministic(small_universe):
    a = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02, target_return=0.15,
    )
    b = compare_risk_metrics(
        small_universe["mu"], small_universe["Sigma"],
        small_universe["returns"], 0.02, target_return=0.15,
    )
    pd.testing.assert_frame_equal(a, b, check_exact=True)