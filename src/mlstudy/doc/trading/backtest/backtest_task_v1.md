
i want to claude to look at codes in src/mlstudy/trading/backtest, and update any if needed to support my backtest functionality i want to have
(0) do a for loop all given market data with initial state = flat
(1) test signal is focusing on mean reverting now, in yield space, zscore (based on mid yield) to decide to signal wants to enter or exit, signal value = expected pnl in bps
(2) try enter if signal wants to enter (zscore> a threshold), and convert ( multiplier * expected pnl in bps - enter cost premium in bps) to acceptable cost in prices delta using dv01
(3) calculate the notional to enter or exit based on hedge ratios in yield space and dv01
(4) check bid offer market is valid, i,e, provided mid price in top book bid offer price
(5) check the liquidity: if signal is buying, check offer prices and sizes from top level to deeper level, until fully filled, calculate the weighted price to buy, and calculate the cost: weighted price - mid; enter position if weighted price - mid <= acceptable cost in prices delta;  like wise for selling
(6) if enters, change state to long or short and update trades tracker, change state to otherwise try_long_but_no_liquidity(if cannot be fully filled),try_long_but_bid_offer_wide (if cost > acceptable cost in prices delta) etc.
(7) signal repeat 3-6 until until enters
(8) tracking pnl 
(9) try taking profit if signal (zscore + signal package yield delta > a soft threshold) or signal package yield delta > a hard threshold; and convert ( multiplier * expected pnl in bps - take profit cost premium in bps) to acceptable cost in prices delta using dv01
(10) calculate notional to exit based on the position
(11) repeat (4)
(12) repeat (5) and the exit if condition ok
(13) change state to take_profit/take_profit_quarantine (if tuning for a gating period after taking profit) and update trades tracker, try_take_profit_but_no_liquidity, try_take_profit_but_bid_offer_wide
(13) try stop loss/max_holding if signal package yield delta > a hard threshold;; and convert ( multiplier * expected pnl in bps - stop loss cost premium in bps) to acceptable cost in prices delta using dv01
(14) repeat (5), but no condition check, force to exit
(15) change state to stop_loss/stop_loss_quarantine(if tuning for a gating period after stopping loss)/max_holding/max_holding_quarantine  and update trades tracker
(16) change state flat if quarantine period passed (if no quarantine tuning, then  quarantine period =0)
(17) always update pnl tracker and trades tracker at each step

can you review my backtester design and generate the claude task