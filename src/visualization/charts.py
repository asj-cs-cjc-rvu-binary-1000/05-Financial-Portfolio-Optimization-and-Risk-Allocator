"""
src/visualization/charts.py
Phase 3.7 – Matrix Visualization (Correlation Heatmap + Eigenvalue Bar)

Plotly-only (approved stack), no seaborn/matplotlib.
Functions return go.Figure, do not call .show() inside – caller decides display/save/Streamlit.

Mathematical:
- Heatmap visualizes Corr_{ij} – off-diagonal near ±1 = near-collinear, near 0 = diversification potential.
  Directly relevant to why some eigenvalues near zero.
- Eigenvalue bar chart visualizes risk budget across independent directions. Steep drop-off = covariance effectively low-rank,
  common feature of equity data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

__all__ = [
    "plot_correlation_heatmap",
    "plot_eigenvalue_spectrum",
    "plot_covariance_heatmap",
    "plot_efficient_frontier",
    "plot_drawdown",
    "plot_var_histogram",
]


def plot_correlation_heatmap(Corr: pd.DataFrame) -> go.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        Corr: (N,N) DataFrame, values in [-1,1], index/columns tickers

    Returns:
        go.Figure: square heatmap, ticker labels both axes, colorscale RdBu centered at 0, range [-1,1],
                   annotations with numeric correlation rounded to 2 decimals

    Title: "Correlation Matrix — Daily Returns"
    """
    if Corr.empty:
        raise ValueError("plot_correlation_heatmap received empty DataFrame")

    if Corr.shape[0] != Corr.shape[1]:
        raise ValueError(f"Correlation must be square, got {Corr.shape}")

    # Ensure values in [-1,1]
    arr = Corr.values
    # Clip for display but keep original for annotations? Use clipped for color scale
    # Annotations from original

    fig = go.Figure(
        data=go.Heatmap(
            z=arr,
            x=list(Corr.columns),
            y=list(Corr.index),
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )

    # Add annotations rounded to 2 decimals
    annotations = []
    for i, row_label in enumerate(Corr.index):
        for j, col_label in enumerate(Corr.columns):
            val = arr[i, j]
            # Choose text color based on background intensity
            # Dark background for strong correlations
            text_color = "white" if abs(val) > 0.5 else "black"
            annotations.append(
                dict(
                    x=col_label,
                    y=row_label,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color=text_color, size=10),
                )
            )

    fig.update_layout(
        title="Correlation Matrix — Daily Returns",
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        width=600,
        height=600,
        annotations=annotations,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),  # Match matrix display
    )

    # Square aspect
    fig.update_layout(
        xaxis=dict(constrain="domain"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def plot_covariance_heatmap(Sigma: pd.DataFrame, title: str = "Covariance Matrix — Daily Returns") -> go.Figure:
    """
    Optional helper: covariance heatmap (not required but useful for debugging).

    Args:
        Sigma: (N,N) covariance DataFrame
        title: chart title

    Returns:
        go.Figure
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=Sigma.values,
            x=list(Sigma.columns),
            y=list(Sigma.index),
            colorscale="Viridis",
            colorbar=dict(title="Covariance"),
            hovertemplate="%{y} vs %{x}: %{z:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Ticker",
        yaxis_title="Ticker",
        width=600,
        height=600,
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def plot_eigenvalue_spectrum(eigen_dict: dict) -> go.Figure:
    """
    Plot eigenvalue spectrum bar chart + cumulative explained variance line.

    Args:
        eigen_dict: dict from eigen_decompose, containing values, explained_variance_ratio, cumulative_explained

    Returns:
        go.Figure: bar chart eigenvalues descending x=PC1..PCN y=eigenvalue,
                   second trace cumulative explained variance on secondary y-axis line+markers

    Title: "Eigenvalue Spectrum and Cumulative Explained Variance"
    """
    values = np.asarray(eigen_dict["values"])
    explained = np.asarray(eigen_dict.get("explained_variance_ratio", values / values.sum()))
    cumulative = np.asarray(eigen_dict.get("cumulative_explained", np.cumsum(explained)))
    N = len(values)

    pc_labels = [f"PC{i+1}" for i in range(N)]

    fig = go.Figure()

    # Bar chart eigenvalues
    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=values,
            name="Eigenvalue",
            marker_color="steelblue",
            yaxis="y",
            text=[f"{v:.4f}" for v in values],
            textposition="auto",
            hovertemplate="%{x}: %{y:.6f}<extra></extra>",
        )
    )

    # Line cumulative explained variance on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cumulative,
            name="Cumulative Explained",
            mode="lines+markers",
            marker=dict(color="indianred", size=8),
            line=dict(color="indianred", width=2),
            yaxis="y2",
            hovertemplate="%{x}: %{y:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Eigenvalue Spectrum and Cumulative Explained Variance",
        xaxis_title="Principal Component",
        yaxis=dict(title="Eigenvalue", side="left"),
        yaxis2=dict(
            title="Cumulative Explained Variance",
            side="right",
            overlaying="y",
            range=[0, 1.05],
            tickformat=".0%",
        ),
        legend=dict(x=0.7, y=0.2),
        template="plotly_white",
        height=500,
        width=700,
        hovermode="x unified",
    )

    return fig

# ---------------------------------------------------------------------------
# Phase 5 Task 5.6 — Risk & Frontier visualizations
# ---------------------------------------------------------------------------


def plot_efficient_frontier(analysis: dict) -> go.Figure:
    """
    Plot the Markowitz efficient frontier with all key portfolios annotated.

    Layers (in plotting order):
        1. Inefficient branch     — dashed gray line
        2. Efficient branch       — solid blue line
        3. Individual assets      — scatter points labeled with ticker
        4. Minimum-volatility     — green triangle marker
        5. Maximum-Sharpe         — red star marker
        6. Equal-weight benchmark — gray circle marker

    Args:
        analysis: dict from analyze_frontier with keys
            efficient_df, inefficient_df, min_var, max_sharpe, equal_weight, assets

    Returns:
        go.Figure. X-axis = annualized volatility, Y-axis = annualized return.
    """
    required = ["efficient_df", "inefficient_df", "min_var", "max_sharpe",
                "equal_weight", "assets"]
    for k in required:
        if k not in analysis:
            raise ValueError(f"plot_efficient_frontier: analysis is missing '{k}'")

    eff = analysis["efficient_df"]
    ineff = analysis["inefficient_df"]
    min_var = analysis["min_var"]
    max_sharpe = analysis["max_sharpe"]
    equal_weight = analysis["equal_weight"]
    assets = analysis["assets"]

    fig = go.Figure()

    # 1. Inefficient branch (plotted first so efficient branch sits on top)
    ineff_ok = ineff[ineff["success"]]
    if not ineff_ok.empty:
        fig.add_trace(go.Scatter(
            x=ineff_ok["volatility"], y=ineff_ok["return"],
            mode="lines",
            name="Inefficient Branch",
            line=dict(color="gray", dash="dash", width=1.5),
            hovertemplate="σ=%{x:.4f}<br>μ=%{y:.4f}<extra>Inefficient</extra>",
        ))

    # 2. Efficient branch
    eff_ok = eff[eff["success"]]
    if not eff_ok.empty:
        fig.add_trace(go.Scatter(
            x=eff_ok["volatility"], y=eff_ok["return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#1f77b4", width=2.5),
            hovertemplate="σ=%{x:.4f}<br>μ=%{y:.4f}<extra>Efficient</extra>",
        ))

    # 3. Individual assets (scatter, one per ticker)
    fig.add_trace(go.Scatter(
        x=assets["sigma"], y=assets["mu"],
        mode="markers+text",
        text=assets["ticker"],
        textposition="top center",
        textfont=dict(size=10, color="#666666"),
        marker=dict(size=9, color="#888888", symbol="circle",
                    line=dict(color="white", width=1)),
        name="Individual Assets",
        hovertemplate="%{text}<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>",
    ))

    # 4. Min-volatility marker
    fig.add_trace(go.Scatter(
        x=[min_var["volatility"]], y=[min_var["return"]],
        mode="markers",
        name="Min Volatility",
        marker=dict(size=16, color="green", symbol="triangle-up",
                    line=dict(color="black", width=1)),
        hovertemplate="Min Vol<br>σ=%{x:.4f}<br>μ=%{y:.4f}<extra></extra>",
    ))

    # 5. Max-Sharpe marker
    fig.add_trace(go.Scatter(
        x=[max_sharpe["volatility"]], y=[max_sharpe["return"]],
        mode="markers",
        name="Max Sharpe",
        marker=dict(size=18, color="red", symbol="star",
                    line=dict(color="black", width=1)),
        hovertemplate=(
            f"Max Sharpe<br>σ=%{{x:.4f}}<br>μ=%{{y:.4f}}"
            f"<br>Sharpe={max_sharpe['sharpe']:.3f}<extra></extra>"
        ),
    ))

    # 6. Equal-weight marker
    fig.add_trace(go.Scatter(
        x=[equal_weight["volatility"]], y=[equal_weight["return"]],
        mode="markers",
        name="Equal Weight",
        marker=dict(size=14, color="gray", symbol="circle",
                    line=dict(color="black", width=1)),
        hovertemplate=(
            f"Equal Weight<br>σ=%{{x:.4f}}<br>μ=%{{y:.4f}}"
            f"<br>Sharpe={equal_weight['sharpe']:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title="Efficient Frontier — Markowitz Mean-Variance",
        xaxis_title="Annualized Volatility (σ)",
        yaxis_title="Annualized Expected Return (μ)",
        template="plotly_white",
        height=560,
        width=900,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#cccccc", borderwidth=1),
        hovermode="closest",
    )
    # Percent formatting on axes for readability
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")

    return fig


def plot_drawdown(
    drawdown_series: pd.Series,
    portfolio_name: str = "Portfolio",
) -> go.Figure:
    """
    Plot the historical drawdown of a portfolio as a filled area chart.

    Y-axis is shown as a percentage (converted from decimal at display time;
    do NOT convert in the input series). The series is expected to already be
    the output of `drawdown_series(returns)`, i.e. signed values ≤ 0.

    Args:
        drawdown_series: pd.Series, DatetimeIndex, values ≤ 0.
        portfolio_name: title suffix.

    Returns:
        go.Figure.
    """
    if not isinstance(drawdown_series, pd.Series):
        raise ValueError("plot_drawdown: drawdown_series must be a pd.Series")
    if drawdown_series.empty:
        raise ValueError("plot_drawdown: drawdown_series is empty")
    if (drawdown_series > 1e-12).any():
        raise ValueError("plot_drawdown: drawdown_series must be ≤ 0 (use src.risk.drawdown_series)")

    pct = drawdown_series * 100.0  # for display only

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pct.index, y=pct.values,
        mode="lines",
        name="Drawdown",
        line=dict(color="firebrick", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(178, 34, 34, 0.25)",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))

    # Mark the maximum drawdown with an annotation
    mdd_idx = pct.idxmin()
    mdd_val = float(pct.min())
    fig.add_trace(go.Scatter(
        x=[mdd_idx], y=[mdd_val],
        mode="markers",
        name="Max Drawdown",
        marker=dict(size=10, color="black", symbol="x"),
        hovertemplate=f"MDD<br>%{{x|%Y-%m-%d}}<br>{mdd_val:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        title=f"Drawdown — {portfolio_name}",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=450,
        width=900,
        hovermode="x unified",
        legend=dict(x=0.02, y=0.05),
    )
    return fig


def plot_var_histogram(
    returns: pd.Series,
    var_value: float,
    portfolio_name: str = "Portfolio",
    confidence_level: float = 0.95,
) -> go.Figure:
    """
    Plot the daily return distribution with the VaR threshold marked.

    The threshold line is drawn at `-var_value` in RETURN space (the quantile
    itself), because `var_value` is a positive loss magnitude (see
    docs/risk_analysis_specification.md Section 2.2). Returns at or below
    that threshold are highlighted as the tail.

    Args:
        returns: pd.Series of daily returns.
        var_value: positive loss magnitude (output of historical_var).
        portfolio_name: title suffix.
        confidence_level: 0 < c < 1, used only for the title and annotation.

    Returns:
        go.Figure.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("plot_var_histogram: returns must be a pd.Series")
    if returns.empty:
        raise ValueError("plot_var_histogram: returns is empty")
    if not np.isfinite(var_value):
        raise ValueError(f"plot_var_histogram: var_value must be finite, got {var_value}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("plot_var_histogram: confidence_level must be in (0, 1)")

    threshold = -float(var_value)  # return-space threshold

    # Two histogram traces: main body and tail
    tail = returns[returns <= threshold]
    body = returns[returns > threshold]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=body.values * 100.0,
        name="Body",
        marker_color="#1f77b4",
        opacity=0.75,
        hovertemplate="%{x:.2f}%<br>count=%{y}<extra>Body</extra>",
    ))
    if not tail.empty:
        fig.add_trace(go.Histogram(
            x=tail.values * 100.0,
            name=f"Tail (≤ VaR, n={len(tail)})",
            marker_color="firebrick",
            opacity=0.85,
            hovertemplate="%{x:.2f}%<br>count=%{y}<extra>Tail</extra>",
        ))

    fig.update_layout(
        title=(
            f"Daily Returns Distribution with VaR Threshold — {portfolio_name} "
            f"({confidence_level:.0%})"
        ),
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        barmode="overlay",
        template="plotly_white",
        height=450,
        width=900,
        legend=dict(x=0.02, y=0.95),
    )

    # Vertical line at threshold
    fig.add_vline(
        x=threshold * 100.0,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"VaR = {var_value*100:.2f}%",
        annotation_position="top",
    )

    return fig

if __name__ == "__main__":
    from pathlib import Path
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        import yfinance
    except ModuleNotFoundError:
        import types
        sys.modules["yfinance"] = types.ModuleType("yfinance")

    from src.data.stock_data import load_prices
    from src.preprocessing.returns import process_returns_pipeline
    from src.linear_algebra.covariance import compute_covariance_matrix, compute_correlation_matrix
    from src.linear_algebra.eigen_analysis import eigen_decompose

    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        processed_dir = repo_root / "data" / "processed"

    csv_files = list(processed_dir.glob("prices_clean_*.csv"))
    if not csv_files:
        print("No processed CSV found")
        raise SystemExit(1)

    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading {latest}")
    prices = load_prices(latest)
    rets, _, _, _ = process_returns_pipeline(prices)

    Sigma = compute_covariance_matrix(rets)
    Corr = compute_correlation_matrix(Sigma)
    eig = eigen_decompose(Sigma)

    fig1 = plot_correlation_heatmap(Corr)
    fig2 = plot_eigenvalue_spectrum(eig)

    print(f"fig1 7x7: {fig1.data[0].z.shape}, type {type(fig1)}")
    print(f"fig2 bars {len(fig2.data[0].y)}, cum ends {fig2.data[1].y[-1]}")

    screenshots_dir = Path("docs/screenshots")
    if not screenshots_dir.exists():
        screenshots_dir = repo_root / "docs" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, name, **kwargs):
        png_path = screenshots_dir / f"{name}.png"
        html_path = screenshots_dir / f"{name}.html"
        # Try kaleido 0.2.1 first (plotly 6.x compatible)
        try:
            fig.write_image(str(png_path), engine="kaleido", **kwargs)
            print(f"Saved {png_path} via kaleido 0.2.1")
            return True
        except Exception as e:
            print(f"kaleido 0.2.1 failed for {name}: {e}")

        # Try default (kaleido v1)
        try:
            fig.write_image(str(png_path), **kwargs)
            print(f"Saved {png_path} via default engine")
            return True
        except Exception as e:
            print(f"default write_image failed for {name}: {e}")

        # Try with chrome auto-install
        try:
            import kaleido
            if hasattr(kaleido, "get_chrome_sync"):
                kaleido.get_chrome_sync()
            fig.write_image(str(png_path), **kwargs)
            print(f"Saved {png_path} after get_chrome_sync")
            return True
        except Exception as e:
            print(f"chrome fallback failed for {name}: {e}")

        # Last resort HTML
        try:
            fig.write_html(str(html_path))
            print(f"Saved fallback HTML {html_path}")
        except Exception as e:
            print(f"HTML fallback also failed: {e}")
        return False

    ok1 = save_fig(fig1, "corr_heatmap", width=600, height=600, scale=2)
    ok2 = save_fig(fig2, "eigen_spectrum", width=700, height=500, scale=2)

    # ---- Phase 5 — frontier, drawdown, VaR histogram (Task 5.6) ----
    from src.optimization import maximum_sharpe_portfolio
    from src.risk import (
        analyze_frontier,
        drawdown_series,
        historical_var,
        portfolio_historical_returns,
    )

    R_F = 0.02
    analysis = analyze_frontier(mu=rets.mean() * 252, Sigma=Sigma * 252,
                                risk_free_rate=R_F, n_points=50)

    fig3 = plot_efficient_frontier(analysis)
    ok3 = save_fig(fig3, "efficient_frontier", width=900, height=560, scale=2)

    # Drawdown + VaR on the max-Sharpe portfolio
    ms = maximum_sharpe_portfolio(rets.mean() * 252, Sigma * 252, R_F)
    R_p_ms = portfolio_historical_returns(ms["weights"], rets)
    fig4 = plot_drawdown(drawdown_series(R_p_ms), portfolio_name="Max Sharpe")
    ok4 = save_fig(fig4, "drawdown_max_sharpe", width=900, height=450, scale=2)

    var_ms = historical_var(R_p_ms, confidence_level=0.95)
    fig5 = plot_var_histogram(R_p_ms, var_ms, portfolio_name="Max Sharpe",
                              confidence_level=0.95)
    ok5 = save_fig(fig5, "var_histogram_max_sharpe", width=900, height=450, scale=2)

    if ok1 and ok2 and ok3 and ok4 and ok5:
        print("All PNGs saved successfully")
    else:
        print("Some PNGs failed - check kaleido version")
    print("Done")