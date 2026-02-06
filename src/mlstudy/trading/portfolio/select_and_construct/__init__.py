"""Unified strategy selection and portfolio construction.

This module consolidates functionality from:
- strategy_pre_selector.py (static universe selection)
- strategy_portfolio_constructor.py (portfolio construction)
- new_issuance/portfolio.py (time-varying universe handling)

Key Components
--------------
**Metrics** (metrics.py):
    Performance and risk metrics: sharpe, max_drawdown, newey_west_tstat, etc.

**Correlation** (correlation.py):
    Correlation and covariance estimation: sample, EWMA, shrinkage, Ledoit-Wolf,
    plus pairwise overlap correlation for non-synchronous data.

**Clustering** (clustering.py):
    Correlation-based clustering for redundancy reduction.

**Weights** (weights.py):
    Portfolio weight computation: equal, inverse vol, risk parity, mean-variance.

**Gates** (gates.py):
    Configuration classes: StandaloneGates, RedundancyGates, UniverseConfig.

**Selector** (selector.py):
    StrategySelector for both static and time-varying universes.

**Constructor** (constructor.py):
    PortfolioConstructor for building and backtesting portfolios.

Example Usage
-------------
Static universe (all strategies share same observation period):

    from mlstudy.trading.portfolio.select_and_construct import (
        StrategySelector,
        PortfolioConstructor,
        StandaloneGates,
        RedundancyGates,
    )

    # Select strategies
    selector = StrategySelector(returns_dict, ann_factor=252)
    report = selector.run(
        standalone=StandaloneGates(min_sharpe=0.3, max_mdd=-0.25),
        redundancy=RedundancyGates(corr_threshold=0.8, keep_per_cluster=2),
    )

    # Construct portfolio
    constructor = PortfolioConstructor.from_series(
        {name: returns_dict[name] for name in report.selected_names}
    )
    weights, returns_df, cov = constructor.construct(weight_method="risk_parity")
    portfolio_returns = constructor.backtest(returns_df, weights)

Time-varying universe (strategies have different active periods):

    from mlstudy.trading.portfolio.select_and_construct import (
        build_returns_panel,
        StrategySelector,
        PortfolioConstructor,
        UniverseConfig,
        StandaloneGates,
        RedundancyGates,
    )

    # Build panel with NaN where instruments are inactive
    panel = build_returns_panel(backtests_dict)

    # Create selector and constructor
    selector = StrategySelector.from_panel(panel, ann_factor=252)
    constructor = PortfolioConstructor.from_panel(panel)

    # Run dynamic backtest
    portfolio_returns, weights_history = constructor.run_dynamic_backtest(
        selector=selector,
        universe_cfg=UniverseConfig(lookback=126, rebalance_freq="W-FRI"),
        standalone=StandaloneGates(min_obs=60, min_sharpe=None),
        redundancy=RedundancyGates(corr_threshold=0.8),
        weight_method="risk_parity",
    )
"""
from __future__ import annotations

# Metrics
from .metrics import (
    annualized_return,
    annualized_vol,
    sharpe,
    max_drawdown,
    worst_rolling_return,
    downside_deviation,
    sortino,
    newey_west_tstat,
    turnover_proxy,
    positive_year_fraction,
)

# Correlation and covariance
from .correlation import (
    CovMethod,
    CorrMethod,
    corr_matrix,
    pairwise_overlap_corr,
    vol_scale,
    sample_cov,
    ewma_cov,
    diag_shrink_cov,
    ledoit_wolf_cov,
    estimate_cov,
    cov_from_pairwise_corr,
    make_psd,
)

# Clustering
from .clustering import (
    correlation_clusters,
    hierarchical_cluster,
    threshold_clusters,
)

# Weights
from .weights import (
    portfolio_vol,
    risk_contributions,
    equal_weights,
    equal_weights_series,
    inverse_vol_weights,
    risk_parity_weights,
    risk_parity_weights_series,
    mean_variance_weights,
    no_trade_band,
)

# Gates and config
from .gates import (
    RankMetric,
    StandaloneGates,
    RedundancyGates,
    UniverseConfig,
)

# Selector
from .selector import (
    SelectorReport,
    align_returns,
    build_returns_panel,
    StrategySelector,
    StrategyPreSelector,  # alias
    UniverseAwarePreSelector,  # alias
)

# Constructor
from .constructor import (
    WeightMethod,
    StrategyBacktest,
    align_strategy_returns,
    PortfolioConstructor,
    StrategyPortfolioConstructor,  # alias
    UniverseAwarePortfolioBacktester,  # alias
)


__all__ = [
    # Metrics
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "max_drawdown",
    "worst_rolling_return",
    "downside_deviation",
    "sortino",
    "newey_west_tstat",
    "turnover_proxy",
    "positive_year_fraction",
    # Correlation/covariance
    "CovMethod",
    "CorrMethod",
    "corr_matrix",
    "pairwise_overlap_corr",
    "vol_scale",
    "sample_cov",
    "ewma_cov",
    "diag_shrink_cov",
    "ledoit_wolf_cov",
    "estimate_cov",
    "cov_from_pairwise_corr",
    "make_psd",
    # Clustering
    "correlation_clusters",
    "hierarchical_cluster",
    "threshold_clusters",
    # Weights
    "portfolio_vol",
    "risk_contributions",
    "equal_weights",
    "equal_weights_series",
    "inverse_vol_weights",
    "risk_parity_weights",
    "risk_parity_weights_series",
    "mean_variance_weights",
    "no_trade_band",
    # Gates/config
    "RankMetric",
    "StandaloneGates",
    "RedundancyGates",
    "UniverseConfig",
    # Selector
    "SelectorReport",
    "align_returns",
    "build_returns_panel",
    "StrategySelector",
    "StrategyPreSelector",
    "UniverseAwarePreSelector",
    # Constructor
    "WeightMethod",
    "StrategyBacktest",
    "align_strategy_returns",
    "PortfolioConstructor",
    "StrategyPortfolioConstructor",
    "UniverseAwarePortfolioBacktester",
]
