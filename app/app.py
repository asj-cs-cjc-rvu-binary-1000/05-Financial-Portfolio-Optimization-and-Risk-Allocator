"""
app/app.py
Phase 6 — Interactive Streamlit Dashboard.

A read-only rendering layer on top of the Phase 4 and Phase 5 pipelines.
No mathematical logic is implemented here — every displayed number is the
output of an existing `src.*` public function.

Launch:
    streamlit run app/app.py
Serves on http://localhost:8501 (forwarded by the devcontainer from Phase 0).

Tabs:
    - Optimization (fully functional in Group B)
    - Risk Analysis (Group C placeholder)
    - Frontier (Group C placeholder)

Sidebar controls (spec §2.1):
    ticker multiselect, date range, risk-free rate, VaR confidence,
    frontier n_points, optional target-return checkbox + slider.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap — MUST come before any `src.*` import.
#
# Streamlit runs this file with the CWD set to `app/`, so the repo root is
# not on sys.path and `from src.* import ...` would raise ModuleNotFoundError.
# Insert the repo root (parent of `app/`) at position 0 before importing
# anything from `src.*`.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.stock_data import DEFAULT_TICKERS, load_prices
from src.preprocessing.returns import (
    compute_expected_returns,
    compute_volatility,
    process_returns_pipeline,
)
from src.linear_algebra.covariance import compute_covariance_matrix
from src.optimization import compare_portfolios
from src.risk import (
    analyze_frontier,
    compare_risk_metrics,
    drawdown_series,
    historical_cvar,
    historical_var,
    maximum_drawdown,
    portfolio_historical_returns,
)
from src.visualization.charts import (
    plot_efficient_frontier,
    plot_drawdown,
    plot_var_histogram,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"

_COLORS = {
    "EqualWeight": "#888888",
    "MinVolatility": "#2ca02c",
    "MaxSharpe": "#d62728",
    "TargetReturn": "#9467bd",
}

_CONFIDENCE_OPTIONS = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
_DEFAULT_CONFIDENCE = "95%"


# ---------------------------------------------------------------------------
# Cached data pipeline (spec §4)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_prices(path: str) -> pd.DataFrame:
    """Load a processed price CSV. Cached on the path string."""
    return load_prices(path)


@st.cache_data(show_spinner=False)
def _returns_full(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily returns on the full price history. Cached on the prices DataFrame."""
    returns, _, _, _ = process_returns_pipeline(prices)
    return returns


# ---------------------------------------------------------------------------
# Formatters (spec §3)
# ---------------------------------------------------------------------------


def _fmt_pct(x: float) -> str:
    """Format a decimal as a percentage with 2 decimals. NaN → '—'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_pct_signed(x: float) -> str:
    """Format a signed decimal as a percentage (e.g., -30.19%)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_sharpe(x: float) -> str:
    """Format Sharpe with 3 decimals. NaN → '—'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.3f}"


def _fmt_weight(x: float) -> str:
    """Format a weight as a decimal with 4 places. NaN → '—'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.4f}"


def _fmt_success(x: bool) -> str:
    return "✅" if x else "❌"


# ---------------------------------------------------------------------------
# Input validation (spec §2.2, §5.1)
# ---------------------------------------------------------------------------

def _target_slider_bounds(mu: pd.Series) -> tuple[float, float]:
    """
    Return (lo, hi) in decimal percent-units for the target-return slider.

    Returns (nan, nan) if mu is empty or constant.
    """
    if mu is None or len(mu) == 0:
        return (float("nan"), float("nan"))
    lo = float(mu.min())
    hi = float(mu.max())
    return (lo, hi)

def _validate_inputs(
    tickers: list[str],
    date_start: pd.Timestamp,
    date_end: pd.Timestamp,
    risk_free_rate: float,
    include_target: bool,
    target_return: float | None,
    mu_min: float,
    mu_max: float,
) -> tuple[list[str], list[str]]:
    """
    Return (errors, warnings).

    - errors block the Run button.
    - warnings are informational only.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if len(tickers) < 2:
        errors.append("Select at least 2 tickers to compute covariance.")
    if date_start >= date_end:
        errors.append("Start date must be before end date.")
    if not np.isfinite(risk_free_rate):
        errors.append("Risk-free rate must be a finite number.")
    if risk_free_rate > 0.15:
        warnings.append(
            f"Risk-free rate {risk_free_rate*100:.1f}% is unusually high — "
            "check the input."
        )
    if include_target and target_return is not None:
        if target_return < mu_min or target_return > mu_max:
            errors.append(
                f"Target return {target_return*100:.2f}% is outside "
                f"[{mu_min*100:.2f}%, {mu_max*100:.2f}%] for the selected assets."
            )

    return errors, warnings


# ---------------------------------------------------------------------------
# Latest CSV discovery
# ---------------------------------------------------------------------------


def _find_latest_csv() -> Path | None:
    """Return the most recently modified processed CSV, or None."""
    if not DATA_DIR.exists():
        return None
    csvs = sorted(DATA_DIR.glob("prices_clean_*.csv"))
    if not csvs:
        return None
    return csvs[-1]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(prices_full: pd.DataFrame) -> dict:
    """
    Render all sidebar controls and return the validated dashboard inputs
    as a dict. See spec §2.1.
    """
    st.sidebar.title("Portfolio Configurator")

    # Ticker multiselect — restricted to the CSV's tickers
    available_tickers = list(prices_full.columns)
    default_tickers = [t for t in DEFAULT_TICKERS if t in available_tickers]

    selected_tickers = st.sidebar.multiselect(
        "Tickers",
        options=available_tickers,
        default=default_tickers,
        help="Select at least 2. Tickers come from the currently loaded CSV.",
    )

    # Date range — default to full CSV
    full_start = prices_full.index[0].date()
    full_end = prices_full.index[-1].date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(full_start, full_end),
        min_value=full_start,
        max_value=full_end,
    )
    # date_input returns a tuple (or a single date if the user is mid-pick)
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        date_start_raw, date_end_raw = date_range
    else:
        date_start_raw = date_end_raw = full_start
    date_start = pd.Timestamp(date_start_raw)
    date_end = pd.Timestamp(date_end_raw)

    # Risk-free rate slider (0.0% – 20.0%, step 0.5%)
    r_f_pct = st.sidebar.slider(
        "Risk-free rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=2.0,
        step=0.5,
        help="Annualized, decimal % — 2.0 means 2%/yr.",
    )
    risk_free_rate = r_f_pct / 100.0

    # VaR/CVaR confidence
    confidence_label = st.sidebar.selectbox(
        "VaR / CVaR confidence",
        options=list(_CONFIDENCE_OPTIONS.keys()),
        index=list(_CONFIDENCE_OPTIONS.keys()).index(_DEFAULT_CONFIDENCE),
    )
    confidence_level = _CONFIDENCE_OPTIONS[confidence_label]

    # Frontier resolution
    n_points = st.sidebar.slider(
        "Frontier resolution (points per branch)",
        min_value=20,
        max_value=100,
        value=50,
        step=10,
    )

    # Optional target return
    include_target = st.sidebar.checkbox("Include Target Return portfolio", value=False)
    target_return: float | None = None
    mu_min = mu_max = 0.0
    if include_target:
        # Dynamic bounds from currently selected assets' mu
        if len(selected_tickers) >= 2:
            subset = prices_full[selected_tickers].loc[date_start:date_end]
            if not subset.empty:
                returns_sub = subset.pct_change().dropna()
                if not returns_sub.empty:
                    mu_sub = compute_expected_returns(returns_sub, annualize=True)
                    mu_min, mu_max = _target_slider_bounds(mu_sub)
        if mu_min < mu_max:
            target_return = st.sidebar.slider(
                "Target return (%)",
                min_value=float(mu_min * 100.0),
                max_value=float(mu_max * 100.0),
                value=float(((mu_min + mu_max) / 2.0) * 100.0),
                step=0.1,
            ) / 100.0
        else:
            st.sidebar.warning("Cannot compute target bounds — select at least 2 tickers.")

    return {
        "selected_tickers": selected_tickers,
        "date_start": date_start,
        "date_end": date_end,
        "risk_free_rate": risk_free_rate,
        "confidence_level": confidence_level,
        "confidence_label": confidence_label,
        "n_points": int(n_points),
        "include_target": include_target,
        "target_return": target_return,
        "mu_min": mu_min,
        "mu_max": mu_max,
    }


# ---------------------------------------------------------------------------
# Data filtering + recomputation (spec amendment — see Group B note)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _filtered_data(
    prices: pd.DataFrame,
    tickers_tuple: tuple[str, ...],
    date_start: str,
    date_end: str,
) -> dict:
    """
    Slice prices to the selected tickers + date range, then recompute
    mu / sigma / Sigma on the filtered returns.
    """
    tickers = list(tickers_tuple)
    subset = prices[tickers].loc[date_start:date_end]
    returns, _, _, _ = process_returns_pipeline(subset)
    mu = compute_expected_returns(returns, annualize=True)
    sigma = compute_volatility(returns, annualize=True)
    Sigma = compute_covariance_matrix(returns, annualize=True)
    return {"returns": returns, "mu": mu, "sigma": sigma, "Sigma": Sigma}


# ---------------------------------------------------------------------------
# Tab 1 — Optimization (Task 6.5)
# ---------------------------------------------------------------------------


def render_optimization_tab(inputs: dict, data: dict) -> None:
    """Comparison table + allocation chart for the selected portfolios."""
    st.subheader("Portfolio comparison")

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", len(inputs["selected_tickers"]))
    c2.metric("Risk-free rate", f"{inputs['risk_free_rate']*100:.1f}%")
    c3.metric("Confidence", inputs["confidence_label"])
    if inputs["include_target"] and inputs["target_return"] is not None:
        c4.metric("Target return", f"{inputs['target_return']*100:.2f}%")
    else:
        c4.metric("Target return", "—")

    with st.spinner("Running optimization…"):
        df = compare_portfolios(
            data["mu"],
            data["Sigma"],
            risk_free_rate=inputs["risk_free_rate"],
            target_return=inputs["target_return"] if inputs["include_target"] else None,
        )

    # Render the comparison table with friendly formatting
    ticker_cols = list(data["mu"].index)
    display = df.copy()
    for c in ("return", "volatility", "variance"):
        display[c] = display[c].apply(_fmt_pct)
    display["sharpe"] = display["sharpe"].apply(_fmt_sharpe)
    display["success"] = display["success"].apply(_fmt_success)
    for t in ticker_cols:
        display[t] = display[t].apply(_fmt_weight)

    # Reorder: metrics first, then weights
    cols = ["return", "volatility", "variance", "sharpe", "success", "n_iter"] + ticker_cols
    display = display[cols]

    st.dataframe(display, use_container_width=True)

    # Allocation bar chart
    st.subheader("Portfolio allocations")
    fig = go.Figure()
    for strategy in df.index:
        if strategy not in _COLORS:
            continue
        fig.add_trace(go.Bar(
            name=strategy,
            x=ticker_cols,
            y=df.loc[strategy, ticker_cols].astype(float).values,
            marker_color=_COLORS[strategy],
        ))
    fig.update_layout(
        barmode="group",
        xaxis_title="Ticker",
        yaxis_title="Weight",
        yaxis_range=[0, 1],
        template="plotly_white",
        height=420,
        legend=dict(x=0.02, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Success / failure panel
    failed = df[~df["success"].astype(bool)]
    if failed.empty:
        st.success("All portfolios solved.")
    else:
        st.warning(f"{len(failed)} portfolio(s) failed to solve:")
        for row_name in failed.index:
            st.write(f"- **{row_name}**: see the `success` column in the table above.")


# ---------------------------------------------------------------------------
# Tab 2 + Tab 3 — Group C placeholders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cached frontier analysis (spec §4)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _analyze_frontier(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    r_f: float,
    n_points: int,
) -> dict:
    """Cached wrapper around src.risk.analyze_frontier."""
    return analyze_frontier(mu, Sigma, risk_free_rate=r_f, n_points=n_points)


# ---------------------------------------------------------------------------
# Tab 2 — Risk Analysis (Task 6.6)
# ---------------------------------------------------------------------------


def render_risk_tab(inputs: dict, data: dict) -> None:
    """Risk comparison table + drawdown chart + VaR histogram (spec §2.4)."""
    st.subheader("Risk comparison")

    # 1. Risk comparison table
    with st.spinner("Computing risk metrics…"):
        risk_df = compare_risk_metrics(
            data["mu"],
            data["Sigma"],
            data["returns"],
            risk_free_rate=inputs["risk_free_rate"],
            target_return=inputs["target_return"] if inputs["include_target"] else None,
            confidence_level=inputs["confidence_level"],
        )

    ticker_cols = list(data["mu"].index)
    display = risk_df.copy()
    for c in ("return", "volatility"):
        display[c] = display[c].apply(_fmt_pct)
    display["sharpe"] = display["sharpe"].apply(_fmt_sharpe)
    display["var_95"] = display["var_95"].apply(_fmt_pct)
    display["cvar_95"] = display["cvar_95"].apply(_fmt_pct)
    display["max_drawdown"] = display["max_drawdown"].apply(_fmt_pct_signed)
    display["success"] = display["success"].apply(_fmt_success)
    for t in ticker_cols:
        display[t] = display[t].apply(_fmt_weight)

    cols = ["return", "volatility", "sharpe", "var_95", "cvar_95",
            "max_drawdown", "success"] + ticker_cols
    st.dataframe(display[cols], use_container_width=True)

    # 2. Portfolio selector
    successful = risk_df[risk_df["success"].astype(bool)]
    if successful.empty:
        st.warning("No successful portfolios to inspect.")
        return

    # Default: portfolio with the highest Sharpe
    default_portfolio = successful["sharpe"].astype(float).idxmax()
    options = list(risk_df.index)
    selected = st.radio(
        "Inspect portfolio",
        options=options,
        index=options.index(default_portfolio),
        horizontal=True,
    )

    row = risk_df.loc[selected]
    if not bool(row["success"]):
        st.warning(f"Portfolio `{selected}` failed to solve — cannot compute "
                   f"historical risk metrics.")
        return

    # 3. Portfolio return series + risk metric cards
    w = pd.Series(
        [float(row[t]) for t in ticker_cols],
        index=ticker_cols,
    )
    try:
        R_p = portfolio_historical_returns(w, data["returns"])
    except Exception as e:
        st.error(f"Failed to compute portfolio return series: {type(e).__name__}: {e}")
        return

    var_value = historical_var(R_p, inputs["confidence_level"])
    cvar_value = historical_cvar(R_p, inputs["confidence_level"])
    mdd_value = maximum_drawdown(R_p)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"VaR ({int(inputs['confidence_level']*100)}%)", _fmt_pct(var_value))
    c2.metric(f"CVaR ({int(inputs['confidence_level']*100)}%)", _fmt_pct(cvar_value))
    c3.metric("Max drawdown", _fmt_pct_signed(mdd_value))

    # 4. Drawdown chart
    st.subheader(f"Drawdown — {selected}")
    try:
        dd = drawdown_series(R_p)
        fig_dd = plot_drawdown(dd, portfolio_name=selected)
        st.plotly_chart(fig_dd, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render drawdown chart: {type(e).__name__}: {e}")

    # 5. VaR histogram
    st.subheader(f"Daily return distribution — {selected}")
    try:
        fig_hist = plot_var_histogram(
            R_p,
            var_value,
            portfolio_name=selected,
            confidence_level=inputs["confidence_level"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render VaR histogram: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Tab 3 — Efficient Frontier (Task 6.6)
# ---------------------------------------------------------------------------


def render_frontier_tab(inputs: dict, data: dict) -> None:
    """Efficient frontier chart + anchor cards + metadata (spec §2.5)."""
    st.subheader("Efficient frontier")

    st.caption(
        f"Frontier computed at r_f = {inputs['risk_free_rate']*100:.1f}%, "
        f"n_points = {inputs['n_points']} per branch."
    )

    with st.spinner(f"Computing frontier ({inputs['n_points']} × 2 points)…"):
        analysis = _analyze_frontier(
            data["mu"], data["Sigma"],
            inputs["risk_free_rate"], inputs["n_points"],
        )

    # 1. Efficient frontier chart (6-layer figure from Phase 5)
    try:
        fig = plot_efficient_frontier(analysis)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render efficient frontier: {type(e).__name__}: {e}")
        return

    # 2. Anchor cards
    st.subheader("Key portfolios")
    c1, c2, c3 = st.columns(3)

    def _anchor_card(col, title: str, anchor: dict, show_sharpe: bool = True):
        col.markdown(f"**{title}**")
        col.metric("Return", _fmt_pct(anchor["return"]))
        col.metric("Volatility", _fmt_pct(anchor["volatility"]))
        if show_sharpe and "sharpe" in anchor:
            col.metric("Sharpe", _fmt_sharpe(anchor["sharpe"]))

    _anchor_card(c1, "Min Volatility", analysis["min_var"], show_sharpe=False)
    _anchor_card(c2, "Max Sharpe", analysis["max_sharpe"])
    _anchor_card(c3, "Equal Weight", analysis["equal_weight"])

    # 3. Metadata table
    st.subheader("Frontier metadata")
    meta = analysis["metadata"]
    meta_display = pd.DataFrame(
        [(k, v) for k, v in meta.items()],
        columns=["Key", "Value"],
    )
    st.dataframe(meta_display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Portfolio Optimizer",
        page_icon="📈",
        layout="wide",
    )
    st.title("Financial Portfolio Optimizer")
    st.caption(
        "A read-only dashboard on top of the Phase 4 (Markowitz) and "
        "Phase 5 (risk) pipelines. All numbers are computed by the "
        "`src.*` modules — nothing is re-derived inside Streamlit."
    )

    # 1. Discover the latest CSV
    csv_path = _find_latest_csv()
    if csv_path is None:
        st.error(
            f"No processed prices found in `{DATA_DIR}`. "
            "Run `python -m src.data.stock_data` from a terminal first."
        )
        return

    # 2. Load prices (cached)
    try:
        prices_full = _load_prices(str(csv_path))
    except Exception as e:
        st.error(f"Failed to load `{csv_path.name}`: {type(e).__name__}: {e}")
        return

    st.caption(
        f"Data: `{csv_path.name}`  "
        f"({prices_full.shape[0]} prices, "
        f"{prices_full.index[0].date()} → {prices_full.index[-1].date()})"
    )

    # 3. Sidebar controls
    inputs = render_sidebar(prices_full)

    # 4. Pre-run validation (spec §2.2, §5.1)
    errors, warnings = _validate_inputs(
        tickers=inputs["selected_tickers"],
        date_start=inputs["date_start"],
        date_end=inputs["date_end"],
        risk_free_rate=inputs["risk_free_rate"],
        include_target=inputs["include_target"],
        target_return=inputs["target_return"],
        mu_min=inputs["mu_min"],
        mu_max=inputs["mu_max"],
    )
    for w in warnings:
        st.sidebar.warning(w)
    if errors:
        for e in errors:
            st.sidebar.error(e)
        st.info(
            "Fix the errors in the sidebar to enable the dashboard. "
            "The current values cannot be used for optimization."
        )
        return

    # 5. Filtered data (recomputed mu / Sigma on the slice)
    try:
        data = _filtered_data(
            prices=prices_full,
            tickers_tuple=tuple(inputs["selected_tickers"]),
            date_start=str(inputs["date_start"].date()),
            date_end=str(inputs["date_end"].date()),
        )
    except Exception as e:
        st.error(f"Failed to filter data: {type(e).__name__}: {e}")
        return

    # 6. Three tabs — each tab's render is isolated (spec §5.2)
    tab_opt, tab_risk, tab_frontier = st.tabs(
        ["Optimization", "Risk Analysis", "Frontier"]
    )
    tabs = [
        (tab_opt, render_optimization_tab, "Optimization"),
        (tab_risk, render_risk_tab, "Risk Analysis"),
        (tab_frontier, render_frontier_tab, "Frontier"),
    ]
    for tab_obj, render_fn, name in tabs:
        with tab_obj:
            try:
                render_fn(inputs, data)
            except Exception as e:
                st.error(f"[{name}] {type(e).__name__}: {e}")
                # Note: no st.stop() — other tabs must still render (spec §5.2)


if __name__ == "__main__":
    main()