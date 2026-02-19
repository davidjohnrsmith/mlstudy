# Detailed Summary: `src/mlstudy/trading/backtest/portfolio`

This document provides a detailed summary of the `src/mlstudy/trading/backtest/portfolio` module, which implements a portfolio-level backtesting framework based on linear programming (LP) optimization.

## 1. Overall Architecture

The module follows a similar structure to the `mean_reversion` backtester, with a clear separation of concerns:

-   **`single_backtest`**: The core backtesting engine for a single portfolio strategy.
-   **`sweep`**: Functionality for running parameter sweeps (assumed to be similar to the `mean_reversion` backtester).
-   **`configs`**: Configuration management for backtests and sweeps.
-   **`data`**: Data loading and preparation.
-   **`parameters`**: Definition of the backtest parameters.

The key difference lies in the core logic within the `single_backtest` module, which is designed for portfolio-level optimization rather than a simple pairwise mean-reversion strategy.

## 2. `single_backtest` Module

This module contains the core logic for the LP-based portfolio backtester.

-   **`engine.py`**: The entry point for a single backtest run. It validates inputs and orchestrates the execution of the backtest loop.
-   **`loop.py`**: The core of the backtester. It implements a bar-by-bar simulation of a portfolio strategy that uses linear programming to optimize trades.
    -   **Signal Gating and Candidate Generation**: At each bar, the backtester generates a set of potential trade candidates. This process involves several steps:
        1.  **Fair Price Gating**: The strategy uses two sets of "fair prices" for each instrument: one for risk-increasing trades and one for risk-decreasing trades. These fair prices are gated by signals like z-scores and ADF p-values, with stricter criteria for risk-increasing trades.
        2.  **Risk Classification**: Each potential trade (buy or sell) is classified as either risk-increasing or risk-decreasing based on the current position in the instrument.
        3.  **Executable Alpha**: For each potential trade, the "executable alpha" is calculated in basis points. This is the difference between the fair price and the market price (ask for buys, bid for sells).
        4.  **Candidate Filtering**: The potential trades are filtered based on various criteria, including whether the fair price is active, the executable alpha is above a certain threshold, there is liquidity in the order book, and the trade satisfies a minimum maturity constraint.
    -   **LP Formulation**: The filtered trade candidates are then fed into a linear programming solver to determine the optimal allocation of capital.
        -   **Objective Function**: The LP solver maximizes the total executable alpha across all trades, weighted by the DV01 of each trade.
        -   **Constraints**: The optimization is subject to a set of constraints:
            -   **Liquidity**: The size of each trade is limited by the available liquidity in the order book.
            -   **Gross DV01**: The total DV01 of all trades in a bar is limited by a gross DV01 cap.
            -   **Position Limits**: The size of each trade is limited by the maximum long and short position limits for each instrument.
            -   **Bucket Constraints**: The total DV01 exposure can be limited per issuer and per maturity bucket.
    -   **LP Solver and Fallback**: The code uses `scipy.optimize.linprog` to solve the LP problem. If `scipy` is not installed or if the solver fails, it falls back to a simpler greedy algorithm that allocates capital to the trades with the highest alpha first.
    -   **Execution Model**: After the LP solver determines the optimal trade sizes (in DV01 terms), the trades are converted to notional quantities, rounded, and then executed by "walking the book" in the L2 order book simulation. The backtester allows for partial fills.
    -   **Hedging**: The backtester also supports hedging. For each executed trade, it can calculate a target hedge position based on pre-defined hedge ratios and execute the net hedge for the bar.
-   **`results.py` and `state.py`**: These files define the data structures and enums used to store the results and state of the backtest, similar to the `mean_reversion` backtester.

## 3. Other Modules

The `configs`, `data`, `parameters`, and `sweep` modules are assumed to have a similar function to their counterparts in the `mean_reversion` backtester, providing a framework for configuring, running, and analyzing backtests and sweeps.

## Bug Review

I have reviewed the code for bugs and have not found any obvious logical errors. The logic is significantly more complex than the `mean_reversion` backtester, which increases the potential for subtle bugs. Here are some key points from the review:

-   **Soft Dependency on Scipy**: The use of a greedy fallback when `scipy` is not available is a good design choice that makes the code more portable.
-   **Error Handling**: The LP solver call is wrapped in a `try...except` block, which makes the backtester more robust to solver failures.
-   **Code Quality**: The code is well-structured and documented, with a clear separation of concerns. The use of Numba for the core loop is a good optimization.
-   **Testing**: Given the complexity of the logic, especially the LP formulation and the hedging, extensive testing with a wide range of scenarios is crucial to ensure the correctness of the backtester.

Overall, the `src/mlstudy/trading/backtest/portfolio` module provides a powerful and flexible framework for backtesting portfolio-level trading strategies.
