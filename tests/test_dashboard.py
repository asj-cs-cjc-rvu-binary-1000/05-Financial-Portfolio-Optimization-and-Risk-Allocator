"""
tests/test_dashboard.py
Phase 6 — Task 6.8: unit tests for the pure helper functions in app/app.py.

Streamlit widgets are not unit-tested — they are visually verified.
Only the pure helpers (formatters, validation, target bounds) are tested.

The `_sys_path_bootstrap` trick: `app/app.py` does `sys.path.insert(...)`
at import time to make `from src.* import ...` work under `streamlit run`.
When pytest imports `app.app`, that bootstrap runs harmlessly (the repo
root is already on sys.path via pytest.ini). The app module is imported
in `import app.app` below — Streamlit is available (it's in requirements),
and the module's `main()` is never invoked at import.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.app as app_module


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def test_fmt_pct_positive():
    assert app_module._fmt_pct(0.1988) == "19.88%"


def test_fmt_pct_zero():
    assert app_module._fmt_pct(0.0) == "0.00%"


def test_fmt_pct_nan_returns_em_dash():
    assert app_module._fmt_pct(float("nan")) == "—"
    assert app_module._fmt_pct(None) == "—"


def test_fmt_pct_signed_preserves_minus():
    assert app_module._fmt_pct_signed(-0.3019) == "-30.19%"


def test_fmt_sharpe_three_decimals():
    assert app_module._fmt_sharpe(1.0088) == "1.009"
    assert app_module._fmt_sharpe(0.9701) == "0.970"


def test_fmt_sharpe_nan_returns_em_dash():
    assert app_module._fmt_sharpe(float("nan")) == "—"


def test_fmt_weight_four_decimals():
    assert app_module._fmt_weight(0.142857) == "0.1429"


def test_fmt_success():
    assert app_module._fmt_success(True) == "✅"
    assert app_module._fmt_success(False) == "❌"


# ---------------------------------------------------------------------------
# _validate_inputs (spec §2.2)
# ---------------------------------------------------------------------------


def _valid_inputs(**overrides):
    base = dict(
        tickers=["A", "B", "C"],
        date_start=pd.Timestamp("2020-01-01"),
        date_end=pd.Timestamp("2024-12-31"),
        risk_free_rate=0.02,
        include_target=False,
        target_return=None,
        mu_min=0.10,
        mu_max=0.20,
    )
    base.update(overrides)
    return base


def test_validate_valid_inputs_pass():
    errors, warnings = app_module._validate_inputs(**_valid_inputs())
    assert errors == []
    assert warnings == []


def test_validate_too_few_tickers_is_error():
    errors, _ = app_module._validate_inputs(**_valid_inputs(tickers=["A"]))
    assert any("at least 2" in e for e in errors)


def test_validate_start_after_end_is_error():
    errors, _ = app_module._validate_inputs(**_valid_inputs(
        date_start=pd.Timestamp("2024-12-31"),
        date_end=pd.Timestamp("2020-01-01"),
    ))
    assert any("Start date must be before" in e for e in errors)


def test_validate_non_finite_r_f_is_error():
    errors, _ = app_module._validate_inputs(**_valid_inputs(risk_free_rate=float("inf")))
    assert any("finite" in e for e in errors)


def test_validate_high_r_f_is_warning_not_error():
    errors, warnings = app_module._validate_inputs(**_valid_inputs(risk_free_rate=0.25))
    assert errors == []
    assert any("unusually high" in w for w in warnings)


def test_validate_target_out_of_range_is_error():
    errors, _ = app_module._validate_inputs(**_valid_inputs(
        include_target=True,
        target_return=0.99,
        mu_min=0.10,
        mu_max=0.20,
    ))
    assert any("outside" in e for e in errors)


def test_validate_target_in_range_passes():
    errors, warnings = app_module._validate_inputs(**_valid_inputs(
        include_target=True,
        target_return=0.15,
        mu_min=0.10,
        mu_max=0.20,
    ))
    assert errors == []
    assert warnings == []


# ---------------------------------------------------------------------------
# _target_slider_bounds
# ---------------------------------------------------------------------------


def test_target_slider_bounds_normal():
    mu = pd.Series([0.10, 0.15, 0.20], index=["A", "B", "C"])
    lo, hi = app_module._target_slider_bounds(mu)
    assert lo == pytest.approx(0.10)
    assert hi == pytest.approx(0.20)


def test_target_slider_bounds_empty_returns_nan():
    lo, hi = app_module._target_slider_bounds(pd.Series([], dtype=float))
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_target_slider_bounds_constant_returns_same_bounds():
    mu = pd.Series([0.15, 0.15, 0.15], index=["A", "B", "C"])
    lo, hi = app_module._target_slider_bounds(mu)
    assert lo == pytest.approx(0.15)
    assert hi == pytest.approx(0.15)