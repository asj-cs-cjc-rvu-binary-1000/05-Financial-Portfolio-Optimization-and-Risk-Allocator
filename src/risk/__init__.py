"""
src/risk/__init__.py
Phase 5 — Risk Analysis & Efficient Frontier.

Exposes the public surface so downstream consumers can do
`from src.risk import historical_var, compare_risk_metrics, ...`.
"""

from src.risk.metrics import (
    drawdown_series,
    maximum_drawdown,
    historical_var,
    historical_cvar,
    portfolio_historical_returns,
)
from src.risk.frontier import efficient_frontier, analyze_frontier
from src.risk.comparison import compare_risk_metrics

__all__ = [
    "drawdown_series",
    "maximum_drawdown",
    "historical_var",
    "historical_cvar",
    "portfolio_historical_returns",
    "efficient_frontier",
    "analyze_frontier",
    "compare_risk_metrics",
]