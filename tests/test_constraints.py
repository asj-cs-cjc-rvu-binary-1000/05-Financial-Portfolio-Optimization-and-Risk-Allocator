"""
tests/test_constraints.py
Phase 4 — Task 4.4: Offline unit tests for weight validation and
constraint helpers.

These tests exercise the contract in docs/optimization_specification.md
Section 3 (Q7: fully invested, long-only, no weight caps) and Section 5.3
(optimizers must post-validate weights via validate_weights).
"""

import numpy as np
import pandas as pd
import pytest

from src.optimization.constraints import (
    budget_constraint,
    long_only_bounds,
    target_return_constraint,
    validate_weights,
)


# ---------------------------------------------------------------------------
# validate_weights — happy path
# ---------------------------------------------------------------------------


def test_validate_weights_equal_portfolio_passes():
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert validate_weights(w) is True


def test_validate_weights_series_with_matching_labels_passes():
    labels = pd.Index(["A", "B", "C"])
    w = pd.Series([0.2, 0.3, 0.5], index=labels)
    assert validate_weights(w, labels=labels) is True


def test_validate_weights_no_labels_passes():
    w = pd.Series([0.2, 0.3, 0.5], index=["X", "Y", "Z"])
    # no labels arg — index not checked
    assert validate_weights(w) is True


def test_validate_weights_corner_solution_passes():
    w = np.array([1.0, 0.0, 0.0])
    assert validate_weights(w) is True


# ---------------------------------------------------------------------------
# validate_weights — failures
# ---------------------------------------------------------------------------


def test_validate_weights_sum_not_one_raises():
    with pytest.raises(ValueError, match="sum\\(w\\)"):
        validate_weights(np.array([0.5, 0.5, 0.5]))  # sums to 1.5


def test_validate_weights_sum_just_out_of_tol_raises():
    w = np.array([0.5, 0.5 + 5e-6])  # diff 5e-6 > tol 1e-6
    with pytest.raises(ValueError, match="sum\\(w\\)"):
        validate_weights(w, tol=1e-6)


def test_validate_weights_sum_just_in_tol_passes():
    w = np.array([0.5, 0.5 + 5e-7])  # diff 5e-7 < tol 1e-6
    assert validate_weights(w, tol=1e-6) is True


def test_validate_weights_negative_raises():
    with pytest.raises(ValueError, match="negative"):
        validate_weights(np.array([-0.1, 0.5, 0.6]))


def test_validate_weights_above_one_raises():
    """
    Verify that a weight exceeding 1 causes rejection.

    Note on test construction: because the weights must sum to ~1 for the
    sum check to pass, any input containing a weight > 1 MUST also contain
    a negative weight (so the total reaches 1). Thus an input that triggers
    only the max-bound check is mathematically impossible. The current
    validate_weights implementation checks min-bound first, so on a vector
    like [0.5, 1.1, -0.6] the "negative weight" message fires before the
    "exceeds 1" message.

    This test asserts that infeasible input is rejected with a ValueError
    whose message clearly identifies the bound violation — whichever bound
    the implementation checks first. If validate_weights is ever refactored
    to reorder the checks, both messages remain acceptable, and this test
    keeps passing.
    """
    with pytest.raises(ValueError, match="negative weight|exceeds 1"):
        validate_weights(np.array([0.5, 1.1, -0.6]))


def test_validate_weights_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        validate_weights(np.array([]))


def test_validate_weights_non_finite_raises():
    with pytest.raises(ValueError, match="non-finite"):
        validate_weights(np.array([0.5, np.nan, 0.5]))


def test_validate_weights_non_1d_raises():
    with pytest.raises(ValueError, match="must be 1-D"):
        validate_weights(np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_validate_weights_label_mismatch_raises():
    labels = pd.Index(["A", "B", "C"])
    w = pd.Series([0.2, 0.3, 0.5], index=["X", "Y", "Z"])
    with pytest.raises(ValueError, match="does not match labels"):
        validate_weights(w, labels=labels)


# ---------------------------------------------------------------------------
# budget_constraint
# ---------------------------------------------------------------------------


def test_budget_constraint_type_is_eq():
    c = budget_constraint()
    assert c["type"] == "eq"
    assert callable(c["fun"])


def test_budget_constraint_satisfied_at_equal_weights():
    c = budget_constraint()
    w = np.array([0.25, 0.25, 0.25, 0.25])
    assert c["fun"](w) == pytest.approx(0.0, abs=1e-15)


def test_budget_constraint_satisfied_at_corner():
    c = budget_constraint()
    w = np.array([1.0, 0.0, 0.0])
    assert c["fun"](w) == pytest.approx(0.0, abs=1e-15)


def test_budget_constraint_returns_minus_0_1_for_sum_0_9():
    c = budget_constraint()
    w = np.array([0.3, 0.3, 0.3])
    assert c["fun"](w) == pytest.approx(-0.1, abs=1e-15)


def test_budget_constraint_returns_plus_0_5_for_sum_1_5():
    c = budget_constraint()
    w = np.array([0.5, 0.5, 0.5])
    assert c["fun"](w) == pytest.approx(0.5, abs=1e-15)


# ---------------------------------------------------------------------------
# long_only_bounds
# ---------------------------------------------------------------------------


def test_long_only_bounds_shape_and_values():
    bounds = long_only_bounds(3)
    assert bounds == [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]


def test_long_only_bounds_n_equals_one():
    assert long_only_bounds(1) == [(0.0, 1.0)]


def test_long_only_bounds_zero_raises():
    with pytest.raises(ValueError, match="n_assets must be int >= 1"):
        long_only_bounds(0)


def test_long_only_bounds_negative_raises():
    with pytest.raises(ValueError, match="n_assets must be int >= 1"):
        long_only_bounds(-1)


# ---------------------------------------------------------------------------
# target_return_constraint
# ---------------------------------------------------------------------------


def test_target_return_constraint_type_is_eq():
    mu = np.array([0.10, 0.15])
    c = target_return_constraint(mu, 0.12)
    assert c["type"] == "eq"
    assert callable(c["fun"])


def test_target_return_constraint_satisfied():
    mu = np.array([0.10, 0.15])
    target = 0.12
    c = target_return_constraint(mu, target)
    w = np.array([0.6, 0.4])  # 0.6·0.10 + 0.4·0.15 = 0.12
    assert c["fun"](w) == pytest.approx(0.0, abs=1e-15)


def test_target_return_constraint_below_target_returns_negative():
    mu = np.array([0.10, 0.15])
    c = target_return_constraint(mu, 0.12)
    w = np.array([0.5, 0.5])  # 0.125 > 0.12, so fun = 0.005 > 0
    assert c["fun"](w) == pytest.approx(0.005, abs=1e-15)


def test_target_return_constraint_above_target_returns_positive():
    mu = np.array([0.10, 0.15])
    c = target_return_constraint(mu, 0.12)
    w = np.array([0.7, 0.3])  # 0.115 < 0.12, so fun = -0.005 < 0
    assert c["fun"](w) == pytest.approx(-0.005, abs=1e-15)


def test_target_return_constraint_accepts_series():
    mu = pd.Series([0.10, 0.15], index=["A", "B"])
    c = target_return_constraint(mu, 0.12)
    w = np.array([0.6, 0.4])
    assert c["fun"](w) == pytest.approx(0.0, abs=1e-15)


def test_target_return_constraint_wrong_w_shape_raises():
    mu = np.array([0.10, 0.15])
    c = target_return_constraint(mu, 0.12)
    with pytest.raises(ValueError, match="shape"):
        c["fun"](np.array([0.5]))


def test_target_return_constraint_non_finite_mu_raises():
    with pytest.raises(ValueError, match="non-finite"):
        target_return_constraint(np.array([0.10, np.nan]), 0.12)


def test_target_return_constraint_non_finite_target_raises():
    with pytest.raises(ValueError, match="target must be finite"):
        target_return_constraint(np.array([0.10, 0.15]), np.inf)