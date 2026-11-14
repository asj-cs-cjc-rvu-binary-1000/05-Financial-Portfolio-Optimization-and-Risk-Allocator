"""
tests/test_benchmarks.py
Phase 4 — Task 4.5: Offline unit tests for the equal-weight benchmark.

Exercises the contract in docs/optimization_specification.md Section 2.4:
    w_eq = (1/N) · 1
and the return-dict shape shared by all Phase 4 portfolios (Section 7).
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.benchmarks import equal_weight_portfolio, equal_weights
from src.optimization.portfolio_metrics import (
    portfolio_return,
    portfolio_sharpe,
    portfolio_variance,
    portfolio_volatility,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_3asset():
    labels = ["A", "B", "C"]
    mu = pd.Series([0.10, 0.15, 0.20], index=labels)
    Sigma = pd.DataFrame(
        [[0.040, 0.010, 0.005],
         [0.010, 0.090, 0.015],
         [0.005, 0.015, 0.160]],
        index=labels, columns=labels,
    )
    return {"mu": mu, "Sigma": Sigma, "labels": labels}


# ---------------------------------------------------------------------------
# equal_weights — the pure builder
# ---------------------------------------------------------------------------


def test_equal_weights_length_and_sum():
    w = equal_weights(["A", "B", "C"])
    assert len(w) == 3
    assert w.sum() == pytest.approx(1.0, abs=1e-15)
    assert (w.values == np.full(3, 1 / 3)).all()


def test_equal_weights_preserves_index_order():
    w = equal_weights(["C", "A", "B"])
    assert list(w.index) == ["C", "A", "B"]


def test_equal_weights_accepts_pd_index():
    idx = pd.Index(["X", "Y"])
    w = equal_weights(idx)
    assert list(w.index) == ["X", "Y"]
    assert len(w) == 2


def test_equal_weights_accepts_pd_series():
    s = pd.Series([0.1, 0.2], index=["P", "Q"])
    w = equal_weights(s)
    assert list(w.index) == ["P", "Q"]
    assert (w.values == np.array([0.5, 0.5])).all()


def test_equal_weights_empty_raises():
    with pytest.raises(ValueError, match="labels is empty"):
        equal_weights([])


# ---------------------------------------------------------------------------
# equal_weight_portfolio — return-dict shape
# ---------------------------------------------------------------------------


REQUIRED_KEYS = {
    "weights", "return", "volatility", "variance",
    "sharpe", "success", "message", "n_iter",
}


def test_return_dict_has_all_required_keys(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    assert set(result.keys()) == REQUIRED_KEYS


def test_weights_is_series_with_mu_labels(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    assert isinstance(result["weights"], pd.Series)
    assert result["weights"].index.equals(toy_3asset["mu"].index)


def test_success_is_always_true(toy_3asset):
    for r_f in [0.02, None, 0.99]:
        r = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], r_f)
        assert r["success"] is True


def test_n_iter_is_zero(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    assert result["n_iter"] == 0


def test_message_mentions_benchmark(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    assert "benchmark" in result["message"].lower()


# ---------------------------------------------------------------------------
# Metric consistency with Task 4.3 primitives
# ---------------------------------------------------------------------------


def test_return_matches_primitive(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    w = result["weights"]
    expected = portfolio_return(w, toy_3asset["mu"])
    assert result["return"] == pytest.approx(expected, abs=1e-15)


def test_variance_matches_primitive(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    w = result["weights"]
    expected = portfolio_variance(w, toy_3asset["Sigma"])
    assert result["variance"] == pytest.approx(expected, abs=1e-15)


def test_volatility_matches_primitive(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    w = result["weights"]
    expected = portfolio_volatility(w, toy_3asset["Sigma"])
    assert result["volatility"] == pytest.approx(expected, abs=1e-15)


def test_sharpe_matches_primitive(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    w = result["weights"]
    expected = portfolio_sharpe(w, toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    assert result["sharpe"] == pytest.approx(expected, abs=1e-15)


def test_volatility_is_sqrt_of_variance(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], 0.02)
    assert result["volatility"] == pytest.approx(
        np.sqrt(result["variance"]), abs=1e-15)


# ---------------------------------------------------------------------------
# risk_free_rate = None
# ---------------------------------------------------------------------------


def test_risk_free_rate_none_gives_sharpe_none(toy_3asset):
    result = equal_weight_portfolio(toy_3asset["mu"], toy_3asset["Sigma"], None)
    assert result["sharpe"] is None
    # All other metrics still computed
    assert isinstance(result["return"], float)
    assert isinstance(result["volatility"], float)


# ---------------------------------------------------------------------------
# Different asset counts
# ---------------------------------------------------------------------------


def test_n_equals_one():
    labels = ["ONLY"]
    mu = pd.Series([0.12], index=labels)
    Sigma = pd.DataFrame([[0.04]], index=labels, columns=labels)
    result = equal_weight_portfolio(mu, Sigma, 0.02)
    assert (result["weights"].values == np.array([1.0])).all()
    assert result["return"] == pytest.approx(0.12, abs=1e-15)
    assert result["variance"] == pytest.approx(0.04, abs=1e-15)
    assert result["volatility"] == pytest.approx(0.2, abs=1e-15)


def test_n_equals_seven():
    labels = [f"T{i}" for i in range(7)]
    mu = pd.Series([0.10] * 7, index=labels)
    Sigma = pd.DataFrame(np.eye(7) * 0.05, index=labels, columns=labels)
    result = equal_weight_portfolio(mu, Sigma, 0.02)
    assert len(result["weights"]) == 7
    assert (result["weights"].values == np.full(7, 1 / 7)).all()
    assert result["weights"].sum() == pytest.approx(1.0, abs=1e-15)


# ---------------------------------------------------------------------------
# Label invariants
# ---------------------------------------------------------------------------


def test_label_mismatch_mu_vs_sigma_raises():
    mu = pd.Series([0.10, 0.12], index=["A", "B"])
    Sigma = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]],
                         index=["A", "C"], columns=["A", "C"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        equal_weight_portfolio(mu, Sigma, 0.02)


def test_numpy_inputs_raise_clear_error():
    mu = np.array([0.10, 0.12])
    Sigma = np.eye(2) * 0.05
    with pytest.raises(ValueError, match="neither mu nor Sigma carries ticker labels"):
        equal_weight_portfolio(mu, Sigma, 0.02)