"""
src/optimization/__init__.py
Phase 4 — Portfolio Optimization Engine.

Exposes the public surface of the optimization package so downstream
consumers (Phase 5 efficient frontier, Phase 6 dashboard, Phase 7 tests)
can do `from src.optimization import portfolio_return, ...` without
importing submodules directly.

Submodules (added in later Phase 4 tasks) must be re-exported here as
they become available:
- Task 4.4: validate_weights, budget_constraint, long_only_bounds,
            target_return_constraint     (from .constraints)
- Task 4.5: equal_weight_portfolio      (from .benchmarks)
- Task 4.6/4.7/4.8: minimum_variance_portfolio,
                    maximum_sharpe_portfolio,
                    target_return_portfolio  (from .optimizer)
- Task 4.9: compare_portfolios          (from .optimization_pipeline)
"""

from src.optimization.portfolio_metrics import (
    portfolio_return,
    portfolio_variance,
    portfolio_volatility,
    portfolio_sharpe,
)
from src.optimization.constraints import (
    validate_weights,
    budget_constraint,
    long_only_bounds,
    target_return_constraint,
)
from src.optimization.benchmarks import (
    equal_weights,
    equal_weight_portfolio,
)

__all__ = [
    "portfolio_return",
    "portfolio_variance",
    "portfolio_volatility",
    "portfolio_sharpe",
    "validate_weights",
    "budget_constraint",
    "long_only_bounds",
    "target_return_constraint",
    "equal_weights",
    "equal_weight_portfolio",
]