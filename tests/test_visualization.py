"""
tests/test_visualization.py
Phase 5 — Task 5.6: offline tests for the three risk visualization functions.

Locks:
- Each function returns a plotly.graph_objects.Figure (never None, never calls .show()).
- The efficient frontier plot has the expected number of layers and labeled anchors.
- The drawdown plot rejects positive values in the drawdown series (a bug in the caller, not the chart).
- The VaR histogram draws a vertical line at the threshold and marks the tail correctly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.charts import (
    plot_efficient_frontier,
    plot_drawdown,
    plot_var_histogram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_analysis():
    """
    Minimal synthetic analysis dict mirroring the structure produced by
    src.risk.analyze_frontier.
    """
    labels = ["A", "B", "C"]

    eff = pd.DataFrame({
        "target_return": np.linspace(0.12, 0.20, 5),
        "return":        np.linspace(0.12, 0.20, 5),
        "volatility":    np.linspace(0.14, 0.30, 5),
        "variance":      np.linspace(0.0196, 0.09, 5),
        "success":       [True] * 5,
        "n_iter":        [5] * 5,
        "A":             [0.5, 0.4, 0.3, 0.2, 0.1],
        "B":             [0.3, 0.3, 0.3, 0.3, 0.3],
        "C":             [0.2, 0.3, 0.4, 0.5, 0.6],
    })
    ineff = eff.copy()
    ineff["target_return"] = np.linspace(0.08, 0.12, 5)
    ineff["return"]        = np.linspace(0.08, 0.12, 5)

    return {
        "efficient_df": eff,
        "inefficient_df": ineff,
        "metadata": {"mu_min": 0.08, "mu_min_var": 0.12, "mu_max": 0.20,
                     "n_points": 5, "n_success_efficient": 5,
                     "n_success_inefficient": 5,
                     "n_failed_efficient": 0, "n_failed_inefficient": 0},
        "min_var":      {"return": 0.12, "volatility": 0.14},
        "max_sharpe":   {"return": 0.18, "volatility": 0.20, "sharpe": 0.80},
        "equal_weight": {"return": 0.15, "volatility": 0.22, "sharpe": 0.59},
        "assets": pd.DataFrame({
            "ticker": labels,
            "mu":     [0.10, 0.15, 0.20],
            "sigma":  [0.20, 0.25, 0.35],
        }),
    }


@pytest.fixture
def synthetic_drawdown():
    idx = pd.date_range("2020-01-01", periods=20)
    vals = np.array([0, 0, -0.02, -0.05, -0.08, -0.10, -0.07, -0.03, 0,
                     0, -0.01, -0.03, -0.06, -0.09, -0.12, -0.15, -0.10,
                     -0.06, -0.02, 0], dtype=float)
    return pd.Series(vals, index=idx)


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(seed=42)
    return pd.Series(rng.normal(0.0005, 0.02, size=500),
                     index=pd.date_range("2020-01-01", periods=500))


# ---------------------------------------------------------------------------
# plot_efficient_frontier
# ---------------------------------------------------------------------------


def test_efficient_frontier_returns_figure(synthetic_analysis):
    fig = plot_efficient_frontier(synthetic_analysis)
    assert isinstance(fig, go.Figure)


def test_efficient_frontier_layer_count(synthetic_analysis):
    """6 layers: inefficient, efficient, assets, min_var, max_sharpe, equal_weight."""
    fig = plot_efficient_frontier(synthetic_analysis)
    assert len(fig.data) == 6


def test_efficient_frontier_layer_names(synthetic_analysis):
    fig = plot_efficient_frontier(synthetic_analysis)
    names = [t.name for t in fig.data]
    for expected in ["Inefficient Branch", "Efficient Frontier", "Individual Assets",
                     "Min Volatility", "Max Sharpe", "Equal Weight"]:
        assert expected in names, f"missing layer {expected!r} in {names!r}"


def test_efficient_frontier_inefficient_is_dashed(synthetic_analysis):
    fig = plot_efficient_frontier(synthetic_analysis)
    ineff = next(t for t in fig.data if t.name == "Inefficient Branch")
    assert ineff.line.dash == "dash"


def test_efficient_frontier_efficient_is_solid(synthetic_analysis):
    fig = plot_efficient_frontier(synthetic_analysis)
    eff = next(t for t in fig.data if t.name == "Efficient Frontier")
    # plotly treats None/undefined as solid; assert either solid or not-dash
    assert eff.line.dash in (None, "solid")


def test_efficient_frontier_missing_key_raises(synthetic_analysis):
    bad = {**synthetic_analysis}
    del bad["assets"]
    with pytest.raises(ValueError, match="missing 'assets'"):
        plot_efficient_frontier(bad)


def test_efficient_frontier_asset_labels(synthetic_analysis):
    fig = plot_efficient_frontier(synthetic_analysis)
    assets = next(t for t in fig.data if t.name == "Individual Assets")
    assert list(assets.text) == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# plot_drawdown
# ---------------------------------------------------------------------------


def test_drawdown_returns_figure(synthetic_drawdown):
    fig = plot_drawdown(synthetic_drawdown, portfolio_name="Test")
    assert isinstance(fig, go.Figure)


def test_drawdown_layer_count(synthetic_drawdown):
    """2 layers: drawdown series + MDD marker."""
    fig = plot_drawdown(synthetic_drawdown, portfolio_name="Test")
    assert len(fig.data) == 2


def test_drawdown_y_values_are_percentages(synthetic_drawdown):
    fig = plot_drawdown(synthetic_drawdown, portfolio_name="Test")
    dd = next(t for t in fig.data if t.name == "Drawdown")
    # Compare min of plotted values against 100 × min of input
    assert float(np.min(dd.y)) == pytest.approx(100.0 * float(synthetic_drawdown.min()), abs=1e-9)


def test_drawdown_title_includes_portfolio_name(synthetic_drawdown):
    fig = plot_drawdown(synthetic_drawdown, portfolio_name="Max Sharpe")
    assert "Max Sharpe" in fig.layout.title.text


def test_drawdown_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        plot_drawdown(pd.Series([], dtype=float))


def test_drawdown_rejects_positive_series():
    bad = pd.Series([0.01, 0.02, -0.01])  # first two are positive → bug in caller
    with pytest.raises(ValueError, match="must be ≤ 0"):
        plot_drawdown(bad)


# ---------------------------------------------------------------------------
# plot_var_histogram
# ---------------------------------------------------------------------------


def test_var_histogram_returns_figure(synthetic_returns):
    var = -float(np.quantile(synthetic_returns.values, 0.05))
    fig = plot_var_histogram(synthetic_returns, var, portfolio_name="Test")
    assert isinstance(fig, go.Figure)


def test_var_histogram_has_threshold_line(synthetic_returns):
    var = -float(np.quantile(synthetic_returns.values, 0.05))
    fig = plot_var_histogram(synthetic_returns, var, portfolio_name="Test")
    # Threshold line is a vline shape
    shapes = fig.layout.shapes or ()
    assert any(getattr(s, "type", None) == "line" for s in shapes), "no threshold line drawn"


def test_var_histogram_two_histogram_traces(synthetic_returns):
    var = -float(np.quantile(synthetic_returns.values, 0.05))
    fig = plot_var_histogram(synthetic_returns, var)
    hist_traces = [t for t in fig.data if isinstance(t, go.Histogram)]
    assert len(hist_traces) == 2
    names = {t.name for t in hist_traces}
    assert "Body" in names
    assert any(n.startswith("Tail") for n in names)


def test_var_histogram_title_includes_confidence(synthetic_returns):
    var = -float(np.quantile(synthetic_returns.values, 0.05))
    fig = plot_var_histogram(synthetic_returns, var, portfolio_name="X",
                             confidence_level=0.99)
    assert "99%" in fig.layout.title.text


def test_var_histogram_rejects_invalid_confidence(synthetic_returns):
    var = -float(np.quantile(synthetic_returns.values, 0.05))
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        plot_var_histogram(synthetic_returns, var, confidence_level=1.5)


def test_var_histogram_rejects_non_finite_var(synthetic_returns):
    with pytest.raises(ValueError, match="must be finite"):
        plot_var_histogram(synthetic_returns, float("nan"))