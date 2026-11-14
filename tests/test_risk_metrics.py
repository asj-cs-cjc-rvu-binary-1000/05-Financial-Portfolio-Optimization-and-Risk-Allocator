"""
tests/test_risk_metrics.py
Phase 5 — Task 5.3: offline unit tests for drawdown, VaR, CVaR, and
portfolio historical return series.

Locks the contracts in docs/risk_analysis_specification.md Sections 2 and 3:
- maximum_drawdown(returns) == drawdown_series(returns).min() (identity invariant)
- VaR and CVaR return positive loss magnitudes
- Positive-tail edge case gives negative VaR
- CVaR >= VaR in magnitude when the tail has >1 observation
- portfolio_historical_returns computes w^T r_t for each day
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import (
    drawdown_series,
    maximum_drawdown,
    historical_var,
    historical_cvar,
    portfolio_historical_returns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_returns_series():
    """A hand-computable return series covering rise and fall."""
    return pd.Series(
        [0.10, 0.05, -0.20, 0.02, -0.10, 0.03],
        index=pd.date_range("2020-01-01", periods=6),
        name="r",
    )


@pytest.fixture
def multi_asset_returns():
    idx = pd.date_range("2020-01-01", periods=5)
    return pd.DataFrame(
        {"A": [0.01, -0.02, 0.03, -0.04, 0.05],
         "B": [-0.01, 0.02, -0.03, 0.04, -0.05]},
        index=idx,
    )


# ---------------------------------------------------------------------------
# drawdown_series
# ---------------------------------------------------------------------------


def test_drawdown_series_values_nonpositive(simple_returns_series):
    dd = drawdown_series(simple_returns_series)
    assert isinstance(dd, pd.Series)
    assert (dd <= 1e-12).all()


def test_drawdown_series_hand_computed(simple_returns_series):
    """
    Returns: [0.10, 0.05, -0.20, 0.02, -0.10, 0.03]
    C_t    : [1.10, 1.155, 0.924, 0.94248, 0.84823, 0.87368]
    Peak   : [1.10, 1.155, 1.155, 1.155, 1.155, 1.155]
    DD     : [0, 0, (0.924/1.155)-1, ..., ...]
    The worst drawdown is at t=4 (index -0.10 after -0.20):
        DD_4 = (0.84823 - 1.155)/1.155 ≈ -0.2656
    """
    dd = drawdown_series(simple_returns_series)
    assert dd.iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert dd.iloc[1] == pytest.approx(0.0, abs=1e-12)
    # Peak at 1.155; trough after -0.20 is 0.924 → drawdown = -0.20
    assert dd.iloc[2] == pytest.approx((0.924 - 1.155) / 1.155, abs=1e-12)
    assert dd.min() < -0.20


def test_drawdown_series_dataframe(multi_asset_returns):
    dd = drawdown_series(multi_asset_returns)
    assert isinstance(dd, pd.DataFrame)
    assert (dd <= 1e-12).all().all()


# ---------------------------------------------------------------------------
# maximum_drawdown
# ---------------------------------------------------------------------------


def test_maximum_drawdown_equals_series_min(simple_returns_series):
    """Identity invariant (spec Section 2.1)."""
    assert maximum_drawdown(simple_returns_series) == pytest.approx(
        drawdown_series(simple_returns_series).min(), abs=1e-15
    )


def test_maximum_drawdown_is_nonpositive(simple_returns_series):
    assert maximum_drawdown(simple_returns_series) <= 0.0


def test_maximum_drawdown_dataframe(multi_asset_returns):
    """DataFrame case: worst column and worst date."""
    val = maximum_drawdown(multi_asset_returns)
    expected = drawdown_series(multi_asset_returns).min().min()
    assert val == pytest.approx(expected, abs=1e-15)


def test_maximum_drawdown_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        maximum_drawdown(pd.Series([], dtype=float))


def test_maximum_drawdown_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        maximum_drawdown(pd.Series([0.01, np.nan, 0.02]))


# ---------------------------------------------------------------------------
# historical_var
# ---------------------------------------------------------------------------


def test_var_positive_loss(simple_returns_series):
    # Returns sorted: [-0.20, -0.10, 0.02, 0.03, 0.05, 0.10]
    # np.quantile uses LINEAR INTERPOLATION by default (numpy's "linear" method):
    # position 0.05 * (6-1) = 0.25 lies 25% between obs[0] and obs[1]:
    #   Q_0.05 = -0.20 + 0.25 * (-0.10 - -0.20) = -0.175
    # VaR = -Q_0.05 = 0.175
    var = historical_var(simple_returns_series, confidence_level=0.95)
    assert var == pytest.approx(0.175, abs=1e-12)
    assert var >= 0.0


def test_var_quantile_hand_computed():
    """
    Series sorted: [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    np.quantile linear interpolation: position 0.05 * (10-1) = 0.45
      lies 45% between obs[0] = -0.03 and obs[1] = -0.02:
      Q_0.05 = -0.03 + 0.45 * (-0.02 - -0.03) = -0.0255
    VaR = -Q_0.05 = 0.0255
    """
    series = pd.Series([-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    var = historical_var(series, confidence_level=0.95)
    assert var == pytest.approx(0.0255, abs=1e-12)


def test_var_default_confidence():
    series = pd.Series(np.linspace(-0.10, 0.10, 101))
    var_default = historical_var(series)
    var_95 = historical_var(series, confidence_level=0.95)
    assert var_default == pytest.approx(var_95)


def test_var_positive_tail_returns_negative():
    """All returns positive → 5% quantile is positive → VaR negative."""
    series = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    var = historical_var(series, confidence_level=0.95)
    assert var < 0.0


def test_var_rejects_invalid_confidence(simple_returns_series):
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        historical_var(simple_returns_series, confidence_level=0.0)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        historical_var(simple_returns_series, confidence_level=1.0)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        historical_var(simple_returns_series, confidence_level=1.5)


def test_var_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        historical_var(pd.Series([0.01, np.nan, 0.02]))


# ---------------------------------------------------------------------------
# historical_cvar
# ---------------------------------------------------------------------------


def test_cvar_greater_than_var_in_magnitude():
    """CVaR >= VaR in magnitude when the tail has >1 observation."""
    rng = np.random.default_rng(seed=42)
    series = pd.Series(rng.normal(0.0, 0.02, size=500))
    var = historical_var(series, confidence_level=0.95)
    cvar = historical_cvar(series, confidence_level=0.95)
    assert cvar >= var - 1e-12


def test_cvar_hand_computed():
    """
    Design: 40 observations with a clean 2-observation tail of extreme losses.
    Sorted tail: [-0.05, -0.04], then [0.01, 0.02, ..., 0.38].
    np.quantile(40 obs, 0.05) = position 1.95 → interpolated between
    obs[1]=-0.04 and obs[2]=0.01, giving threshold 0.0075.
    Tail observations <= 0.0075: {-0.05, -0.04}.
    CVaR = -mean(-0.05, -0.04) = 0.045.
    """
    tail = [-0.05, -0.04]
    rest = [0.01 * (i + 1) for i in range(38)]
    series = pd.Series(tail + rest)
    cvar = historical_cvar(series, confidence_level=0.95)
    assert cvar == pytest.approx(0.045, abs=1e-12)


def test_cvar_small_tail_falls_back_to_var_with_warning():
    """
    Design: 10 observations with the two smallest both -0.10, so
    np.quantile(arr, 0.05) = -0.10 exactly. Only 1 observation is
    strictly <= that threshold after deduplication (both -0.10 values
    are in the tail, so 2 observations — wait, we need exactly 1).

    Simpler: use a fixture where the tail has only 1 observation.
    10 obs at 5% quantile = position 0.45. If obs[0] is -0.10 and
    obs[1] is +0.02, threshold = -0.10 + 0.45*0.12 = -0.046.
    Tail = arr[arr <= -0.046] = {-0.10} → 1 observation → fallback fires.
    CVaR = -threshold = -(-0.046) = 0.046.
    """
    series = pd.Series([-0.10, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
    with pytest.warns(RuntimeWarning, match="only 1 observation"):
        cvar = historical_cvar(series, confidence_level=0.95)
    # Fallback returns -Q_0.05; verify against historical_var for consistency
    expected_var = historical_var(series, confidence_level=0.95)
    assert cvar == pytest.approx(expected_var, abs=1e-12)
    # And confirm the value numerically
    assert cvar == pytest.approx(0.046, abs=1e-12)


def test_cvar_default_confidence_matches_explicit():
    series = pd.Series(np.linspace(-0.10, 0.10, 101))
    assert historical_cvar(series) == pytest.approx(
        historical_cvar(series, confidence_level=0.95)
    )


# ---------------------------------------------------------------------------
# portfolio_historical_returns
# ---------------------------------------------------------------------------


def test_portfolio_returns_linear_combination(multi_asset_returns):
    w = pd.Series([0.6, 0.4], index=multi_asset_returns.columns)
    R_p = portfolio_historical_returns(w, multi_asset_returns)
    expected = 0.6 * multi_asset_returns["A"] + 0.4 * multi_asset_returns["B"]
    pd.testing.assert_series_equal(R_p, expected, check_names=False, check_exact=True)


def test_portfolio_returns_label_mismatch_raises(multi_asset_returns):
    w = pd.Series([0.6, 0.4], index=["X", "Y"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        portfolio_historical_returns(w, multi_asset_returns)


def test_portfolio_returns_length_mismatch_raises(multi_asset_returns):
    w = np.array([0.5, 0.3, 0.2])  # 3 weights, 2 assets
    with pytest.raises(ValueError, match="2 columns"):
        portfolio_historical_returns(w, multi_asset_returns)


def test_portfolio_returns_preserves_index(multi_asset_returns):
    w = pd.Series([0.5, 0.5], index=multi_asset_returns.columns)
    R_p = portfolio_historical_returns(w, multi_asset_returns)
    assert R_p.index.equals(multi_asset_returns.index)
    assert R_p.name == "portfolio_return"


def test_portfolio_returns_accepts_ndarray():
    idx = pd.date_range("2020-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [-0.01, 0.00, 0.01]}, index=idx)
    R_p = portfolio_historical_returns(np.array([0.5, 0.5]), returns)
    expected = 0.5 * returns["A"] + 0.5 * returns["B"]
    pd.testing.assert_series_equal(R_p, expected, check_names=False, check_exact=True)