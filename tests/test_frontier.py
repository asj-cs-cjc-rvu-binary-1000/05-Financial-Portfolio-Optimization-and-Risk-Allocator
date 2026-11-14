"""
tests/test_frontier.py
Phase 5 — Task 5.4: offline unit tests for the efficient frontier generator.

Locks the contract in docs/risk_analysis_specification.md Section 4.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.frontier import efficient_frontier


REQUIRED_COLS = ["target_return", "return", "volatility", "variance",
                 "success", "n_iter"]


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


# ---------------------------------------------------------------------------
# Shape and schema
# ---------------------------------------------------------------------------


def test_returns_three_tuple(three_asset):
    out = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=10)
    assert len(out) == 3
    eff, ineff, meta = out
    assert isinstance(eff, pd.DataFrame)
    assert isinstance(ineff, pd.DataFrame)
    assert isinstance(meta, dict)


def test_shapes_default_50(three_asset):
    eff, ineff, _ = efficient_frontier(three_asset["mu"], three_asset["Sigma"])
    assert eff.shape[0] == 50
    assert ineff.shape[0] == 50


def test_shapes_custom(three_asset):
    eff, ineff, _ = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=20)
    assert eff.shape[0] == 20
    assert ineff.shape[0] == 20


def test_columns_include_metrics_and_tickers(three_asset):
    eff, ineff, _ = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=5)
    expected = REQUIRED_COLS + three_asset["labels"]
    assert list(eff.columns) == expected
    assert list(ineff.columns) == expected


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_efficient_endpoint_is_mu_max(three_asset):
    eff, _, meta = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=5)
    assert eff["target_return"].iloc[-1] == pytest.approx(meta["mu_max"], abs=1e-12)
    assert eff["target_return"].iloc[0] == pytest.approx(meta["mu_min_var"], abs=1e-12)


def test_inefficient_endpoint_is_mu_min(three_asset):
    _, ineff, meta = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=5)
    assert ineff["target_return"].iloc[0] == pytest.approx(meta["mu_min"], abs=1e-12)
    assert ineff["target_return"].iloc[-1] == pytest.approx(meta["mu_min_var"], abs=1e-12)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_keys(three_asset):
    _, _, meta = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=5)
    for k in ["mu_min", "mu_min_var", "mu_max", "n_points",
              "n_success_efficient", "n_success_inefficient",
              "n_failed_efficient", "n_failed_inefficient"]:
        assert k in meta


def test_metadata_boundaries(three_asset):
    _, _, meta = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=5)
    assert meta["mu_min"] == pytest.approx(float(three_asset["mu"].min()))
    assert meta["mu_max"] == pytest.approx(float(three_asset["mu"].max()))
    assert meta["mu_min"] <= meta["mu_min_var"] <= meta["mu_max"]


def test_all_targets_succeed_on_fixture(three_asset):
    eff, ineff, meta = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=10)
    assert meta["n_failed_efficient"] == 0
    assert meta["n_failed_inefficient"] == 0


# ---------------------------------------------------------------------------
# Frontier properties
# ---------------------------------------------------------------------------


def test_efficient_returns_are_monotone(three_asset):
    """Target returns on the efficient branch increase monotonically."""
    eff, _, _ = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=20)
    achieved = eff["return"].to_numpy()
    assert np.all(np.diff(achieved) >= -1e-6)


def test_weights_sum_to_one_when_successful(three_asset):
    eff, _, _ = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=10)
    tickers = three_asset["labels"]
    for _, row in eff.iterrows():
        if row["success"]:
            assert row[tickers].sum() == pytest.approx(1.0, abs=1e-6)


def test_target_achieved_within_tolerance(three_asset):
    eff, _, _ = efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=10)
    for _, row in eff.iterrows():
        if row["success"]:
            assert abs(row["return"] - row["target_return"]) <= 1e-6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_n_points_below_2(three_asset):
    with pytest.raises(ValueError, match="n_points must be int >= 2"):
        efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=1)


def test_rejects_non_integer_n_points(three_asset):
    with pytest.raises(ValueError, match="n_points must be int"):
        efficient_frontier(three_asset["mu"], three_asset["Sigma"], n_points=5.0)


def test_label_mismatch_raises():
    mu = pd.Series([0.10, 0.15], index=["A", "B"])
    Sigma = pd.DataFrame(np.eye(2) * 0.05, index=["A", "C"], columns=["A", "C"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        efficient_frontier(mu, Sigma)

# ---------------------------------------------------------------------------
# Task 5.5 — analyze_frontier
# ---------------------------------------------------------------------------


from src.risk.frontier import analyze_frontier


def test_analyze_returns_required_keys(three_asset):
    result = analyze_frontier(three_asset["mu"], three_asset["Sigma"], 0.02, n_points=5)
    for k in ["efficient_df", "inefficient_df", "metadata",
              "min_var", "max_sharpe", "equal_weight", "assets"]:
        assert k in result


def test_analyze_min_var_has_required_keys(three_asset):
    result = analyze_frontier(three_asset["mu"], three_asset["Sigma"], 0.02, n_points=5)
    for k in ["return", "volatility", "weights"]:
        assert k in result["min_var"]
    # min_var portfolio should have the lowest vol among the three anchors
    assert result["min_var"]["volatility"] <= result["equal_weight"]["volatility"] + 1e-8


def test_analyze_max_sharpe_has_required_keys(three_asset):
    result = analyze_frontier(three_asset["mu"], three_asset["Sigma"], 0.02, n_points=5)
    for k in ["return", "volatility", "sharpe", "weights"]:
        assert k in result["max_sharpe"]
    # max_sharpe portfolio should have the highest Sharpe among the three anchors
    assert result["max_sharpe"]["sharpe"] >= result["equal_weight"]["sharpe"] - 1e-8


def test_analyze_assets_dataframe_shape(three_asset):
    result = analyze_frontier(three_asset["mu"], three_asset["Sigma"], 0.02, n_points=5)
    assert list(result["assets"].columns) == ["ticker", "mu", "sigma"]
    assert len(result["assets"]) == 3
    assert list(result["assets"]["ticker"]) == three_asset["labels"]
    assert (result["assets"]["sigma"] > 0).all()


def test_analyze_frontier_dataframes_match_efficient_frontier(three_asset):
    """analyze_frontier's DataFrames must equal efficient_frontier's output."""
    direct_eff, direct_ineff, direct_meta = efficient_frontier(
        three_asset["mu"], three_asset["Sigma"], n_points=5
    )
    result = analyze_frontier(three_asset["mu"], three_asset["Sigma"], 0.02, n_points=5)
    pd.testing.assert_frame_equal(result["efficient_df"], direct_eff)
    pd.testing.assert_frame_equal(result["inefficient_df"], direct_ineff)
    assert result["metadata"] == direct_meta