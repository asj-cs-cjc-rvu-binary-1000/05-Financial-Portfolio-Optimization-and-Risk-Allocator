"""
tests/test_optimizer.py
Phase 4 — Task 4.6 (min-variance) portion.
Tasks 4.7 and 4.8 will append additional test classes.

Exercises the contract in docs/optimization_specification.md:
- Section 2.1: min wᵀΣw s.t. 1ᵀw=1, 0≤w≤1.
- Section 5.3: return-dict contract + post-validation downgrade.
- Section 5.4: pre-solver-exception failure dict.
- Section 5.5: non-finite / shape / label rejection.
"""

import numpy as np
import pandas as pd
import pytest

import src.optimization.optimizer as opt_mod
from src.optimization.optimizer import minimum_variance_portfolio
from src.optimization.portfolio_metrics import (
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analytic_2asset():
    """
    2-asset universe where the closed-form unconstrained min-variance
    portfolio has BOTH weights in [0, 1], so the long-only constraint
    does not bind and the optimizer's answer must match the analytic
    oracle.

    Σ = [[0.04, 0.01],
         [0.01, 0.09]]
    Closed form: w* = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1).
    """
    labels = ["A", "B"]
    mu = pd.Series([0.10, 0.15], index=labels)
    Sigma = pd.DataFrame(
        [[0.04, 0.01],
         [0.01, 0.09]],
        index=labels, columns=labels,
    )
    return {"labels": labels, "mu": mu, "Sigma": Sigma}


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


REQUIRED_KEYS = {
    "weights", "return", "volatility", "variance",
    "sharpe", "success", "message", "n_iter",
}


# ---------------------------------------------------------------------------
# Analytic oracle (unconstrained long-only non-binding)
# ---------------------------------------------------------------------------


def test_matches_analytic_2asset_oracle(analytic_2asset):
    mu = analytic_2asset["mu"]
    Sigma = analytic_2asset["Sigma"]
    Sigma_inv = np.linalg.inv(Sigma.to_numpy())
    ones = np.ones(2)
    w_star = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

    # Both weights are positive on this universe — confirm before asserting
    assert w_star[0] > 0 and w_star[1] > 0, "oracle inactive: try another Σ"

    result = minimum_variance_portfolio(mu, Sigma, risk_free_rate=0.02)
    assert result["success"] is True
    np.testing.assert_allclose(
        result["weights"].to_numpy(), w_star, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Constraint satisfaction
# ---------------------------------------------------------------------------


def test_weights_sum_to_one(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert result["success"] is True
    assert result["weights"].sum() == pytest.approx(1.0, abs=1e-6)


def test_weights_in_unit_interval(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    w = result["weights"].to_numpy()
    assert w.min() >= -1e-6
    assert w.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Return-dict contract
# ---------------------------------------------------------------------------


def test_return_dict_has_all_required_keys(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert set(result.keys()) >= REQUIRED_KEYS


def test_weights_is_series_with_mu_labels(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert isinstance(result["weights"], pd.Series)
    assert result["weights"].index.equals(three_asset["mu"].index)


def test_n_iter_positive(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert result["success"] is True
    assert result["n_iter"] > 0


# ---------------------------------------------------------------------------
# Metrics recomputation (not from scipy result.fun)
# ---------------------------------------------------------------------------


def test_variance_recomputed_from_weights(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    w = result["weights"]
    expected = portfolio_variance(w, three_asset["Sigma"])
    assert result["variance"] == pytest.approx(expected, abs=1e-10)


def test_return_recomputed_from_weights(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    w = result["weights"]
    expected = portfolio_return(w, three_asset["mu"])
    assert result["return"] == pytest.approx(expected, abs=1e-10)


def test_volatility_recomputed_from_weights(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    w = result["weights"]
    expected = portfolio_volatility(w, three_asset["Sigma"])
    assert result["volatility"] == pytest.approx(expected, abs=1e-10)


def test_volatility_equals_sqrt_variance(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert result["volatility"] == pytest.approx(
        np.sqrt(result["variance"]), abs=1e-12
    )


# ---------------------------------------------------------------------------
# risk_free_rate handling (spec Section 5.3)
# ---------------------------------------------------------------------------


def test_sharpe_none_when_risk_free_rate_none(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=None
    )
    assert result["success"] is True
    assert result["sharpe"] is None


def test_sharpe_populated_when_risk_free_rate_given(three_asset):
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert result["success"] is True
    assert isinstance(result["sharpe"], float)


# ---------------------------------------------------------------------------
# Design invariant: objective routes through portfolio_variance
# ---------------------------------------------------------------------------


def test_objective_routes_through_portfolio_variance(monkeypatch, three_asset):
    """
    Design invariant: the objective function must call
    src.optimization.portfolio_metrics.portfolio_variance (imported
    into optimizer.py), not inline w @ Sigma @ w.
    """
    calls = {"count": 0}
    original = opt_mod.portfolio_variance

    def counting(w, Sigma):
        calls["count"] += 1
        return original(w, Sigma)

    monkeypatch.setattr(opt_mod, "portfolio_variance", counting)
    minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    # SLSQP will call the objective dozens of times.
    assert calls["count"] > 1


# ---------------------------------------------------------------------------
# Failure cases
# ---------------------------------------------------------------------------


def test_solver_raising_returns_failure_dict(monkeypatch, three_asset):
    """Spec Section 5.4: scipy raises before producing x."""
    def exploding_minimize(*args, **kwargs):
        raise RuntimeError("simulated solver crash")

    monkeypatch.setattr(opt_mod, "minimize", exploding_minimize)
    result = minimum_variance_portfolio(
        three_asset["mu"], three_asset["Sigma"], risk_free_rate=0.02
    )
    assert result["success"] is False
    assert result["weights"] is None
    assert np.isnan(result["return"])
    assert np.isnan(result["variance"])
    assert np.isnan(result["volatility"])
    assert result["n_iter"] == 0
    assert "simulated solver crash" in result["message"]


def test_non_finite_mu_raises():
    mu = pd.Series([0.10, np.nan], index=["A", "B"])
    Sigma = pd.DataFrame(np.eye(2) * 0.05, index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="non-finite"):
        minimum_variance_portfolio(mu, Sigma, risk_free_rate=0.02)


def test_non_finite_sigma_raises():
    mu = pd.Series([0.10, 0.15], index=["A", "B"])
    Sigma = pd.DataFrame([[0.04, 0.01], [0.01, np.nan]],
                         index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError, match="non-finite"):
        minimum_variance_portfolio(mu, Sigma, risk_free_rate=0.02)


def test_label_mismatch_raises():
    mu = pd.Series([0.10, 0.15], index=["A", "B"])
    Sigma = pd.DataFrame(np.eye(2) * 0.05, index=["A", "C"], columns=["A", "C"])
    with pytest.raises(AssertionError, match="Label mismatch"):
        minimum_variance_portfolio(mu, Sigma, risk_free_rate=0.02)

# ---------------------------------------------------------------------------
# Task 4.7 — maximum-Sharpe optimizer tests
# ---------------------------------------------------------------------------


from src.optimization.optimizer import maximum_sharpe_portfolio
from src.optimization.benchmarks import equal_weight_portfolio
from src.optimization.portfolio_metrics import portfolio_sharpe


class TestMaximumSharpe:
    """Max-Sharpe optimizer: contract, constraint, and Q3 negative case."""

    def test_max_sharpe_beats_equal_weight(self, three_asset):
        """The whole point of max-Sharpe is to beat the equal-weight benchmark."""
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r_f = 0.02
        result = maximum_sharpe_portfolio(mu, Sigma, r_f)
        assert result["success"] is True
        eq = equal_weight_portfolio(mu, Sigma, r_f)
        assert result["sharpe"] >= eq["sharpe"] - 1e-8

    def test_weights_sum_to_one(self, three_asset):
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert result["success"] is True
        assert result["weights"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_in_unit_interval(self, three_asset):
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        w = result["weights"].to_numpy()
        assert w.min() >= -1e-6
        assert w.max() <= 1.0 + 1e-6

    def test_return_dict_has_all_required_keys(self, three_asset):
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert set(result.keys()) >= REQUIRED_KEYS

    def test_weights_is_series_with_mu_labels(self, three_asset):
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert isinstance(result["weights"], pd.Series)
        assert result["weights"].index.equals(three_asset["mu"].index)

    def test_n_iter_positive(self, three_asset):
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert result["success"] is True
        assert result["n_iter"] > 0

    def test_sharpe_recomputed_from_weights(self, three_asset):
        mu, Sigma, r_f = three_asset["mu"], three_asset["Sigma"], 0.02
        result = maximum_sharpe_portfolio(mu, Sigma, r_f)
        expected = portfolio_sharpe(result["weights"], mu, Sigma, r_f)
        assert result["sharpe"] == pytest.approx(expected, abs=1e-10)

    def test_risk_free_rate_is_required(self, three_asset):
        with pytest.raises(TypeError):
            maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"])  # noqa: E112

    def test_q3_negative_sharpe_returns_success_true(self):
        """
        Q3: when every long-only portfolio has negative Sharpe, the
        optimizer still returns the true maximum with success=True
        and a message explaining the situation.
        """
        labels = ["A", "B"]
        mu = pd.Series([0.01, 0.02], index=labels)  # tiny returns
        Sigma = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.09]],
            index=labels, columns=labels,
        )
        r_f = 0.99  # absurdly high — no portfolio can beat this
        result = maximum_sharpe_portfolio(mu, Sigma, r_f)
        assert result["success"] is True
        assert result["sharpe"] is not None
        assert result["sharpe"] < 0
        assert "negative" in result["message"].lower()

    def test_zero_vol_sentinel_allows_solver_to_converge(self, monkeypatch, three_asset):
        """
        Q4: the objective catches ZeroDivisionError and returns the sentinel.
        We simulate a scenario where portfolio_sharpe raises on the very
        first objective call (which would happen for w0 if w0's variance
        were zero) — the optimizer must not crash.
        """
        call_count = {"n": 0}
        original = opt_mod.portfolio_sharpe

        def flaky_sharpe(w, mu, Sigma, r_f):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ZeroDivisionError("simulated degenerate weights")
            return original(w, mu, Sigma, r_f)

        monkeypatch.setattr(opt_mod, "portfolio_sharpe", flaky_sharpe)
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert result["success"] is True
        assert call_count["n"] > 1

    def test_design_invariant_objective_uses_portfolio_sharpe(self, monkeypatch, three_asset):
        """The objective must call portfolio_sharpe (imported into optimizer.py)."""
        calls = {"n": 0}
        original = opt_mod.portfolio_sharpe

        def counting(w, mu, Sigma, r_f):
            calls["n"] += 1
            return original(w, mu, Sigma, r_f)

        monkeypatch.setattr(opt_mod, "portfolio_sharpe", counting)
        maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert calls["n"] > 1

    def test_solver_raising_returns_failure_dict(self, monkeypatch, three_asset):
        def exploding_minimize(*args, **kwargs):
            raise RuntimeError("simulated solver crash")

        monkeypatch.setattr(opt_mod, "minimize", exploding_minimize)
        result = maximum_sharpe_portfolio(three_asset["mu"], three_asset["Sigma"], 0.02)
        assert result["success"] is False
        assert result["weights"] is None
        assert np.isnan(result["sharpe"])
        assert result["n_iter"] == 0
        assert "simulated solver crash" in result["message"]

# ---------------------------------------------------------------------------
# Task 4.8 — target-return optimizer tests
# ---------------------------------------------------------------------------


from scipy.optimize import OptimizeResult
from src.optimization.optimizer import target_return_portfolio


class TestTargetReturn:
    """Target-return min-variance optimizer: oracles, Q1/Q2, and 4.3 tolerance."""

    # ------------------------------------------------------------------
    # Analytic oracles
    # ------------------------------------------------------------------

    def test_at_min_variance_return_matches_min_variance_portfolio(self, analytic_2asset):
        """
        Strongest frontier identity: at target = min_variance_return,
        the target-return optimizer must produce the same weights as
        the min-variance optimizer.
        """
        mu = analytic_2asset["mu"]
        Sigma = analytic_2asset["Sigma"]
        mv = minimum_variance_portfolio(mu, Sigma)
        r = target_return_portfolio(mu, Sigma, target_return=mv["return"])
        assert r["success"] is True
        np.testing.assert_allclose(
            r["weights"].to_numpy(), mv["weights"].to_numpy(), atol=1e-5
        )

    def test_at_mu_max_returns_corner(self, analytic_2asset):
        """At target = mu.max(), the corner solution (all on highest-μ asset) wins."""
        mu = analytic_2asset["mu"]
        Sigma = analytic_2asset["Sigma"]
        target = float(mu.max())
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is True
        # mu = [0.10, 0.15], so mu.max() is at index 1 → w = [0, 1]
        expected = np.array([0.0, 1.0])
        np.testing.assert_allclose(r["weights"].to_numpy(), expected, atol=1e-5)

    def test_at_mu_min_returns_lowest_asset(self, analytic_2asset):
        """At target = mu.min(), the lowest-μ asset dominates."""
        mu = analytic_2asset["mu"]
        Sigma = analytic_2asset["Sigma"]
        target = float(mu.min())
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is True
        expected = np.array([1.0, 0.0])
        np.testing.assert_allclose(r["weights"].to_numpy(), expected, atol=1e-5)

    # ------------------------------------------------------------------
    # Target achievability
    # ------------------------------------------------------------------

    def test_midrange_target_achieved(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target = float(mu.mean())  # somewhere between mu.min and mu.max
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is True
        assert abs(r["return"] - target) <= 1e-6

    # ------------------------------------------------------------------
    # Constraint satisfaction
    # ------------------------------------------------------------------

    def test_weights_sum_to_one(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target = float(mu.mean())
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is True
        assert r["weights"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_in_unit_interval(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target = float(mu.mean())
        r = target_return_portfolio(mu, Sigma, target_return=target)
        w = r["weights"].to_numpy()
        assert w.min() >= -1e-6
        assert w.max() <= 1.0 + 1e-6

    # ------------------------------------------------------------------
    # Return-dict contract
    # ------------------------------------------------------------------

    def test_return_dict_has_all_required_keys(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r = target_return_portfolio(mu, Sigma, target_return=float(mu.mean()))
        assert set(r.keys()) >= REQUIRED_KEYS

    def test_weights_is_series_with_mu_labels(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r = target_return_portfolio(mu, Sigma, target_return=float(mu.mean()))
        assert isinstance(r["weights"], pd.Series)
        assert r["weights"].index.equals(mu.index)

    def test_n_iter_positive(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r = target_return_portfolio(mu, Sigma, target_return=float(mu.mean()))
        assert r["success"] is True
        assert r["n_iter"] > 0

    # ------------------------------------------------------------------
    # risk_free_rate handling
    # ------------------------------------------------------------------

    def test_sharpe_none_when_risk_free_rate_none(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r = target_return_portfolio(mu, Sigma, target_return=float(mu.mean()))
        assert r["sharpe"] is None

    def test_sharpe_populated_when_risk_free_rate_given(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r = target_return_portfolio(
            mu, Sigma, target_return=float(mu.mean()), risk_free_rate=0.02
        )
        assert isinstance(r["sharpe"], float)

    # ------------------------------------------------------------------
    # Q1 / Q2 range check
    # ------------------------------------------------------------------

    def test_q1_below_min_returns_failure(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target = float(mu.min()) - 0.05
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is False
        assert r["weights"] is None
        assert "below mu.min" in r["message"]

    def test_q2_above_max_returns_failure(self, three_asset):
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target = float(mu.max()) + 0.05
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is False
        assert r["weights"] is None
        assert "above mu.max" in r["message"]

    # ------------------------------------------------------------------
    # Achievability tolerance (spec 4.3)
    # ------------------------------------------------------------------

    def test_tolerance_downgrades_success(self, monkeypatch, three_asset):
        """
        Spec 4.3: if the solver reports success but |w @ mu - target| > 1e-6,
        downgrade to success=False.
        """
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target = float(mu.mean())

        def fake_minimize(*args, **kwargs):
            return OptimizeResult(
                x=np.array([0.1, 0.1, 0.8]),  # does not satisfy w @ mu = target
                success=True,
                message="fake solver success",
                nit=5,
                fun=0.0,
            )

        monkeypatch.setattr(opt_mod, "minimize", fake_minimize)
        r = target_return_portfolio(mu, Sigma, target_return=target)
        assert r["success"] is False
        assert "differs from target" in r["message"]

    # ------------------------------------------------------------------
    # Design invariant: objective uses portfolio_variance
    # ------------------------------------------------------------------

    def test_objective_uses_portfolio_variance(self, monkeypatch, three_asset):
        calls = {"n": 0}
        original = opt_mod.portfolio_variance

        def counting(w, Sigma):
            calls["n"] += 1
            return original(w, Sigma)

        monkeypatch.setattr(opt_mod, "portfolio_variance", counting)
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        target_return_portfolio(mu, Sigma, target_return=float(mu.mean()))
        assert calls["n"] > 1

    # ------------------------------------------------------------------
    # Failure case
    # ------------------------------------------------------------------

    def test_solver_raising_returns_failure_dict(self, monkeypatch, three_asset):
        def exploding_minimize(*args, **kwargs):
            raise RuntimeError("simulated solver crash")

        monkeypatch.setattr(opt_mod, "minimize", exploding_minimize)
        mu = three_asset["mu"]
        Sigma = three_asset["Sigma"]
        r = target_return_portfolio(mu, Sigma, target_return=float(mu.mean()))
        assert r["success"] is False
        assert r["weights"] is None
        assert np.isnan(r["return"])
        assert r["n_iter"] == 0
        assert "simulated solver crash" in r["message"]