"""
tests/test_portfolio_metrics.py
Phase 4 — Task 4.3: Offline unit tests for portfolio metric calculations.

These tests exercise the contract in docs/optimization_specification.md:
- Correct R_p = wᵀμ, σ_p² = wᵀΣw, σ_p = √(σ_p²), S_p = (R_p − r_f)/σ_p.
- Label mismatch raises AssertionError.
- Non-finite mu/Sigma raises ValueError.
- Zero-volatility Sharpe raises ZeroDivisionError (Q4).
- portfolio_variance routes through quadratic_form (design-invariant test).
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.portfolio_metrics import (
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    portfolio_sharpe,
)


# ---------------------------------------------------------------------------
# Fixtures / reference data
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_2asset():
    """
    2-asset universe with a known analytic answer.
    Sigma[[0.04, 0.01], [0.01, 0.09]], w = [0.6, 0.4], mu = [0.10, 0.15].
    wᵀΣw = 0.6²·0.04 + 2·0.6·0.4·0.01 + 0.4²·0.09
         = 0.0144   + 0.0048            + 0.0144
         = 0.0336
    wᵀμ   = 0.6·0.10 + 0.4·0.15 = 0.06 + 0.06 = 0.12
    σ_p   = √0.0336 ≈ 0.183303
    S_p(r_f=0.02) = (0.12 - 0.02) / √0.0336 ≈ 0.545545
    """
    labels = ["A", "B"]
    mu = pd.Series([0.10, 0.15], index=labels)
    Sigma = pd.DataFrame(
        [[0.04, 0.01],
         [0.01, 0.09]],
        index=labels, columns=labels,
    )
    w = pd.Series([0.6, 0.4], index=labels)
    return {"w": w, "mu": mu, "Sigma": Sigma, "labels": labels}


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_portfolio_return_known_value(toy_2asset):
    r = portfolio_return(toy_2asset["w"], toy_2asset["mu"])
    assert isinstance(r, float)
    assert r == pytest.approx(0.12, abs=1e-12)


def test_portfolio_variance_known_value(toy_2asset):
    v = portfolio_variance(toy_2asset["w"], toy_2asset["Sigma"])
    assert isinstance(v, float)
    assert v == pytest.approx(0.0336, abs=1e-12)


def test_portfolio_volatility_known_value(toy_2asset):
    s = portfolio_volatility(toy_2asset["w"], toy_2asset["Sigma"])
    assert s == pytest.approx(np.sqrt(0.0336), abs=1e-12)


def test_portfolio_sharpe_known_value(toy_2asset):
    sr = portfolio_sharpe(toy_2asset["w"], toy_2asset["mu"],
                          toy_2asset["Sigma"], 0.02)
    assert sr == pytest.approx((0.12 - 0.02) / np.sqrt(0.0336), abs=1e-12)


def test_portfolio_sharpe_negative_allowed(toy_2asset):
    """Q3: negative Sharpe is a valid numerical result, not an error."""
    sr = portfolio_sharpe(toy_2asset["w"], toy_2asset["mu"],
                          toy_2asset["Sigma"], 0.99)
    assert sr < 0.0


# ---------------------------------------------------------------------------
# Equivalence: numpy array inputs give identical results
# ---------------------------------------------------------------------------


def test_numpy_inputs_match_series_inputs(toy_2asset):
    w = toy_2asset["w"].to_numpy()
    mu = toy_2asset["mu"].to_numpy()
    Sigma = toy_2asset["Sigma"].to_numpy()
    assert portfolio_return(w, mu) == pytest.approx(
        portfolio_return(toy_2asset["w"], toy_2asset["mu"]), abs=1e-12)
    assert portfolio_variance(w, Sigma) == pytest.approx(
        portfolio_variance(toy_2asset["w"], toy_2asset["Sigma"]), abs=1e-12)


# ---------------------------------------------------------------------------
# Label-alignment guards
# ---------------------------------------------------------------------------


def test_label_mismatch_w_vs_mu_raises(toy_2asset):
    w_reindexed = toy_2asset["w"].reindex(["B", "A"])  # wrong order
    with pytest.raises(AssertionError, match="Label mismatch"):
        portfolio_return(w_reindexed, toy_2asset["mu"])


def test_label_mismatch_w_vs_sigma_raises(toy_2asset):
    w_wrong = pd.Series([0.6, 0.4], index=["X", "Y"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        portfolio_variance(w_wrong, toy_2asset["Sigma"])


# ---------------------------------------------------------------------------
# Non-finite guards
# ---------------------------------------------------------------------------


def test_non_finite_mu_raises(toy_2asset):
    mu_bad = toy_2asset["mu"].copy()
    mu_bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        portfolio_return(toy_2asset["w"], mu_bad)


def test_non_finite_sigma_raises(toy_2asset):
    Sigma_bad = toy_2asset["Sigma"].copy()
    Sigma_bad.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        portfolio_variance(toy_2asset["w"], Sigma_bad)


# ---------------------------------------------------------------------------
# Dimension guards
# ---------------------------------------------------------------------------


def test_wrong_length_w_raises(toy_2asset):
    """
    A weight vector with wrong ticker labels is rejected by the
    label-alignment guard (AssertionError) before any dimension check.
    Labels are the more fundamental invariant — silent reindexing is
    forbidden by docs/optimization_specification.md Section 1.
    """
    w_bad = pd.Series([0.5], index=["A"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        portfolio_return(w_bad, toy_2asset["mu"])


def test_non_square_sigma_raises(toy_2asset):
    """
    A non-square Sigma whose columns disagree with w raises a label
    mismatch (AssertionError) from the alignment guard. The alignment
    check runs before the square-shape check because a column/row count
    disagreement with w is fundamentally a labeling problem. This test
    also documents the check ordering: labels first, then shape.
    """
    Sigma_bad = pd.DataFrame(
        np.zeros((2, 3)),
        index=["A", "B"], columns=["A", "B", "C"],
    )
    with pytest.raises(AssertionError, match="Label mismatch"):
        portfolio_variance(toy_2asset["w"], Sigma_bad)


# ---------------------------------------------------------------------------
# Q4: zero-volatility Sharpe
# ---------------------------------------------------------------------------


def test_zero_volatility_sharpe_raises(toy_2asset):
    w_zero = pd.Series([0.0, 0.0], index=toy_2asset["labels"])
    with pytest.raises(ZeroDivisionError, match="degenerate"):
        portfolio_sharpe(w_zero, toy_2asset["mu"], toy_2asset["Sigma"], 0.02)


# ---------------------------------------------------------------------------
# Design-invariant: portfolio_variance MUST call quadratic_form
# ---------------------------------------------------------------------------


def test_portfolio_variance_routes_through_quadratic_form(monkeypatch, toy_2asset):
    """
    Design-invariant test: proves portfolio_variance routes through
    src.linear_algebra.matrix_ops.quadratic_form instead of inlining w @ Sigma @ w.
    If we monkeypatch quadratic_form to raise, portfolio_variance must raise.
    """
    import src.optimization.portfolio_metrics as pm

    def exploding_quadratic_form(*args, **kwargs):
        raise RuntimeError("quadratic_form was called")

    monkeypatch.setattr(pm, "quadratic_form", exploding_quadratic_form)
    with pytest.raises(RuntimeError, match="quadratic_form was called"):
        portfolio_variance(toy_2asset["w"], toy_2asset["Sigma"])


def test_portfolio_variance_rejects_nonsymmetric_sigma(toy_2asset):
    """quadratic_form asserts symmetry — portfolio_variance should inherit that."""
    Sigma_bad = toy_2asset["Sigma"].copy()
    Sigma_bad.iloc[0, 1] = 0.0999  # break symmetry
    with pytest.raises(ValueError, match="symmetric"):
        portfolio_variance(toy_2asset["w"], Sigma_bad)
