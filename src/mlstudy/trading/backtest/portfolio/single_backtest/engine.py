"""High-level entry point for the LP portfolio backtester.

Validates inputs, unpacks the config, calls the loop, and wraps raw
arrays into :class:`PortfolioBacktestResults`.
"""

from __future__ import annotations

import logging

import numpy as np

from mlstudy.trading.backtest.common.single_backtest.engine import ensure_f64, validate_l2_shapes
from mlstudy.trading.backtest.portfolio.configs.backtest_config import PortfolioBacktestConfig
from mlstudy.trading.backtest.portfolio.single_backtest.loop import lp_portfolio_loop, LoopState
from mlstudy.trading.backtest.portfolio.single_backtest.results import PortfolioBacktestResults

logger = logging.getLogger(__name__)


def _validate(
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    dv01: np.ndarray,
    fair_price: np.ndarray,
    zscore: np.ndarray,
    adf_p_value: np.ndarray,
    tradable: np.ndarray,
    pos_limits_long: np.ndarray,
    pos_limits_short: np.ndarray,
    max_trade_notional_inc: np.ndarray,
    max_trade_notional_dec: np.ndarray,
    qty_step: np.ndarray,
    hedge_qty_step: np.ndarray,
    hedge_bid_px: np.ndarray,
    hedge_bid_sz: np.ndarray,
    hedge_ask_px: np.ndarray,
    hedge_ask_sz: np.ndarray,
    hedge_mid_px: np.ndarray,
    hedge_dv01: np.ndarray,
    hedge_ratios: np.ndarray,
) -> None:
    """Shape and value checks (fail fast)."""
    T, B, L = validate_l2_shapes(bid_px, bid_sz, ask_px, ask_sz, mid_px)

    for name, arr, expected in [
        ("dv01", dv01, (T, B)),
        ("fair_price", fair_price, (T, B)),
        ("zscore", zscore, (T, B)),
        ("adf_p_value", adf_p_value, (T, B)),
    ]:
        if arr.shape != expected:
            raise ValueError(f"{name} shape {arr.shape} != expected {expected}")

    for name, arr, expected_len in [
        ("tradable", tradable, B),
        ("pos_limits_long", pos_limits_long, B),
        ("pos_limits_short", pos_limits_short, B),
        ("max_trade_notional_inc", max_trade_notional_inc, B),
        ("max_trade_notional_dec", max_trade_notional_dec, B),
        ("qty_step", qty_step, B),
    ]:
        if len(arr) != expected_len:
            raise ValueError(f"{name} length {len(arr)} != expected {expected_len}")

    # Hedge arrays — must be all-None or all-provided
    hedge_arrs = [hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz,
                  hedge_mid_px, hedge_dv01, hedge_ratios]
    n_none = sum(a is None for a in hedge_arrs)
    if 0 < n_none < len(hedge_arrs):
        raise ValueError(
            "Hedge arrays must be all-None or all-provided"
        )
    if n_none == 0:
        T_h, H, L_h = validate_l2_shapes(
            hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz,
            hedge_mid_px, label="hedge_",
        )
        if T_h != T:
            raise ValueError(
                f"hedge T={T_h} != instrument T={T}"
            )
        if hedge_dv01.shape != (T, H):
            raise ValueError(
                f"hedge_dv01 shape {hedge_dv01.shape} != expected {(T, H)}"
            )
        if hedge_ratios.shape != (T, B, H):
            raise ValueError(
                f"hedge_ratios shape {hedge_ratios.shape} != expected {(T, B, H)}"
            )
        if hedge_qty_step is not None and len(hedge_qty_step) != H:
            raise ValueError(
                f"hedge_qty_step length {len(hedge_qty_step)} != expected H={H}"
            )


def run_backtest(
    *,
    # -- Market L2 --
    bid_px: np.ndarray,
    bid_sz: np.ndarray,
    ask_px: np.ndarray,
    ask_sz: np.ndarray,
    mid_px: np.ndarray,
    # -- Risk --
    dv01: np.ndarray,
    # -- Signals --
    fair_price: np.ndarray,
    zscore: np.ndarray,
    adf_p_value: np.ndarray,
    # -- Static meta --
    tradable: np.ndarray,
    pos_limits_long: np.ndarray,
    pos_limits_short: np.ndarray,
    max_trade_notional_inc: np.ndarray,
    max_trade_notional_dec: np.ndarray,
    # -- Meta --
    maturity: np.ndarray,
    issuer_bucket: np.ndarray,
    maturity_bucket: np.ndarray,
    # -- Bucket caps --
    issuer_dv01_caps: np.ndarray,
    mat_bucket_dv01_caps: np.ndarray,
    # -- Hedge arrays (all-None or all-provided) --
    hedge_bid_px: np.ndarray = None,
    hedge_bid_sz: np.ndarray = None,
    hedge_ask_px: np.ndarray = None,
    hedge_ask_sz: np.ndarray = None,
    hedge_mid_px: np.ndarray = None,
    hedge_dv01: np.ndarray = None,
    hedge_ratios: np.ndarray = None,
    # -- Instrument IDs --
    instrument_ids: list[str] = None,
    # -- Per-instrument qty_step --
    qty_step: np.ndarray = None,     # (B,)
    # -- Per-hedge qty_step --
    hedge_qty_step: np.ndarray = None,  # (H,) or None
    # -- Config --
    cfg: PortfolioBacktestConfig = None,
    # -- Context --
    datetimes: np.ndarray = None,
    # -- Chunked state --
    initial_state: LoopState | None = None,
    return_final_state: bool = False,
) -> PortfolioBacktestResults | tuple[PortfolioBacktestResults, LoopState]:
    """Run an LP portfolio backtest.

    Parameters
    ----------
    bid_px, bid_sz, ask_px, ask_sz : (T, B, L)
        L2 order book for the instrument universe.
    mid_px : (T, B)
        Instrument mid prices.
    dv01 : (T, B)
        DV01 per unit par notional.
    fair_price : (T, B)
        Model fair price per instrument.
    zscore : (T, B)
        Z-score signal per instrument.
    adf_p_value : (T, B)
        ADF test p-value per instrument.
    tradable : (B,)
        Boolean/int tradable mask.
    pos_limits_long, pos_limits_short : (B,)
        Position limits per instrument.
    maturity : (T, B) or (B,)
        Years to maturity per instrument.
    issuer_bucket : (B,)
        Issuer bucket labels per instrument.
    maturity_bucket : (T, B) or (B,)
        Maturity bucket labels per instrument.
    issuer_dv01_caps : (n_issuers,)
        Max absolute DV01 per issuer bucket.
    mat_bucket_dv01_caps : (n_buckets,)
        Max absolute DV01 per maturity bucket.
    hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz : (T, H, L)
        L2 order book for hedge instruments.
    hedge_mid_px : (T, H)
        Hedge instrument mid prices.
    hedge_dv01 : (T, H)
        Hedge instrument DV01.
    hedge_ratios : (T, B, H)
        Hedge ratios per instrument per hedge.
    instrument_ids : list[str]
        Instrument ID strings matching B dimension order.
    cfg : PortfolioBacktestConfig
        Scalar parameters.
    datetimes : (T,)
        Bar timestamps for DataFrame output.

    Returns
    -------
    PortfolioBacktestResults
    """
    T, B, L = bid_px.shape
    H = hedge_mid_px.shape[1] if hedge_mid_px is not None and hedge_mid_px.ndim >= 2 else 0
    logger.info(
        "run_backtest: T=%d, B=%d, L=%d, H=%d, initial_state=%s",
        T, B, L, H, initial_state is not None,
    )

    _validate(
        bid_px, bid_sz, ask_px, ask_sz, mid_px,
        dv01, fair_price, zscore, adf_p_value,
        tradable, pos_limits_long, pos_limits_short,
        max_trade_notional_inc, max_trade_notional_dec,
        qty_step, hedge_qty_step,
        hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz,
        hedge_mid_px, hedge_dv01, hedge_ratios,
    )

    B = mid_px.shape[1]
    if len(instrument_ids) != B:
        raise ValueError(
            f"instrument_ids length {len(instrument_ids)} != B={B} "
            f"(must match the instrument dimension of market data arrays)"
        )

    # Convert to float64
    f_bid_px = ensure_f64(bid_px)
    f_bid_sz = ensure_f64(bid_sz)
    f_ask_px = ensure_f64(ask_px)
    f_ask_sz = ensure_f64(ask_sz)
    f_mid_px = ensure_f64(mid_px)
    f_dv01 = ensure_f64(dv01)
    f_fair = ensure_f64(fair_price)
    f_zscore = ensure_f64(zscore)
    f_adf = ensure_f64(adf_p_value)
    f_hedge_bid_px = ensure_f64(hedge_bid_px) if hedge_bid_px is not None else None
    f_hedge_bid_sz = ensure_f64(hedge_bid_sz) if hedge_bid_sz is not None else None
    f_hedge_ask_px = ensure_f64(hedge_ask_px) if hedge_ask_px is not None else None
    f_hedge_ask_sz = ensure_f64(hedge_ask_sz) if hedge_ask_sz is not None else None
    f_hedge_mid_px = ensure_f64(hedge_mid_px) if hedge_mid_px is not None else None
    f_hedge_dv01 = ensure_f64(hedge_dv01) if hedge_dv01 is not None else None
    f_hedge_ratios = ensure_f64(hedge_ratios) if hedge_ratios is not None else None

    raw = lp_portfolio_loop(
        f_bid_px, f_bid_sz, f_ask_px, f_ask_sz, f_mid_px,
        f_dv01, f_fair, f_zscore, f_adf,
        tradable, pos_limits_long, pos_limits_short,
        max_trade_notional_inc, max_trade_notional_dec,
        maturity, issuer_bucket, maturity_bucket,
        # Config scalars
        gross_dv01_cap=float(cfg.gross_dv01_cap),
        issuer_dv01_caps=issuer_dv01_caps,
        mat_bucket_dv01_caps=mat_bucket_dv01_caps,
        top_k=int(cfg.top_k),
        z_inc=float(cfg.z_inc),
        p_inc=float(cfg.p_inc),
        z_dec=float(cfg.z_dec),
        p_dec=float(cfg.p_dec),
        alpha_thr_inc=float(cfg.alpha_thr_inc),
        alpha_thr_dec=float(cfg.alpha_thr_dec),
        max_levels=int(cfg.max_levels),
        haircut=float(cfg.haircut),
        qty_step=qty_step,
        min_qty_trade=float(cfg.min_qty_trade),
        min_fill_ratio=float(cfg.min_fill_ratio),
        cooldown_bars=int(cfg.cooldown_bars),
        cooldown_mode=int(cfg.cooldown_mode),
        min_maturity_inc=float(cfg.min_maturity_inc),
        initial_capital=float(cfg.initial_capital),
        use_greedy=bool(cfg.use_greedy),
        hedge_bid_px=f_hedge_bid_px,
        hedge_bid_sz=f_hedge_bid_sz,
        hedge_ask_px=f_hedge_ask_px,
        hedge_ask_sz=f_hedge_ask_sz,
        hedge_mid_px=f_hedge_mid_px,
        hedge_dv01=f_hedge_dv01,
        hedge_ratios=f_hedge_ratios,
        hedge_qty_step=hedge_qty_step,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )

    if return_final_state:
        final_state = raw[-1]
        raw = raw[:-1]

    results = PortfolioBacktestResults.from_loop_output(
        raw,
        datetimes=datetimes,
        close_time=cfg.close_time if cfg.close_time != "none" else None,
        mid_px=f_mid_px,
        hedge_mid_px=f_hedge_mid_px,
        hedge_bid_px=f_hedge_bid_px[:, :, 0] if f_hedge_bid_px is not None else None,
        hedge_ask_px=f_hedge_ask_px[:, :, 0] if f_hedge_ask_px is not None else None,
        hedge_ratios=f_hedge_ratios,
        dv01=f_dv01,
        hedge_dv01=f_hedge_dv01,
        instrument_ids=instrument_ids,
    )

    logger.info(
        "run_backtest done: n_trades=%d, final_equity=%.2f",
        results.n_trades, results.equity[-1] if len(results.equity) > 0 else 0.0,
    )

    if return_final_state:
        return results, final_state
    return results


def run_backtest_chunked(
    *,
    data_chunks,
    cfg: PortfolioBacktestConfig,
) -> PortfolioBacktestResults:
    """Run a backtest over sequential data chunks, stitching results.

    Parameters
    ----------
    data_chunks : Iterable[dict[str, Any]]
        Each dict contains the keyword arguments for ``run_backtest``
        (market data arrays + datetimes + instrument_ids).
    cfg : PortfolioBacktestConfig
        Scalar parameters for the backtest.

    Returns
    -------
    PortfolioBacktestResults
    """
    state = None
    chunk_results = []
    cumulative_T = 0

    for chunk_idx, chunk_data in enumerate(data_chunks):
        result, state = run_backtest(
            **chunk_data,
            cfg=cfg,
            initial_state=state,
            return_final_state=True,
        )
        chunk_results.append((result, cumulative_T))
        cumulative_T += len(result.equity)
        logger.info(
            "Chunk %d: T=%d, trades=%d, cumulative_T=%d",
            chunk_idx, len(result.equity), result.n_trades, cumulative_T,
        )

    if not chunk_results:
        raise ValueError("data_chunks yielded no chunks")

    n_chunks = len(chunk_results)
    total_trades = sum(r.n_trades for r, _ in chunk_results)
    logger.info(
        "Stitching %d chunks: total_T=%d, total_trades=%d",
        n_chunks, cumulative_T, total_trades,
    )

    if len(chunk_results) == 1:
        return chunk_results[0][0]

    # Stitch per-bar arrays
    first = chunk_results[0][0]

    positions = np.concatenate([r.positions for r, _ in chunk_results], axis=0)
    cash = np.concatenate([r.cash for r, _ in chunk_results])
    equity = np.concatenate([r.equity for r, _ in chunk_results])
    pnl = np.concatenate([r.pnl for r, _ in chunk_results])
    gross_pnl = np.concatenate([r.gross_pnl for r, _ in chunk_results])
    codes = np.concatenate([r.codes for r, _ in chunk_results])
    n_trades_bar = np.concatenate([r.n_trades_bar for r, _ in chunk_results])
    cooldown = np.concatenate([r.cooldown for r, _ in chunk_results])
    hedge_positions = np.concatenate([r.hedge_positions for r, _ in chunk_results], axis=0)
    hedge_pnl = np.concatenate([r.hedge_pnl for r, _ in chunk_results])
    instrument_position_mtm = np.concatenate([r.instrument_position_mtm for r, _ in chunk_results])
    hedge_position_mtm = np.concatenate([r.hedge_position_mtm for r, _ in chunk_results])
    instrument_cash_mtm = np.concatenate([r.instrument_cash_mtm for r, _ in chunk_results])
    hedge_cash_mtm = np.concatenate([r.hedge_cash_mtm for r, _ in chunk_results])
    portfolio_mtm = np.concatenate([r.portfolio_mtm for r, _ in chunk_results])
    instrument_cost = np.concatenate([r.instrument_cost for r, _ in chunk_results])
    hedge_cost_bar = np.concatenate([r.hedge_cost_bar for r, _ in chunk_results])
    portfolio_cost = np.concatenate([r.portfolio_cost for r, _ in chunk_results])

    # Stitch per-trade arrays with bar index offset
    tr_bars = []
    tr_instruments = []
    tr_sides = []
    tr_qty_reqs = []
    tr_qty_fills = []
    tr_dv01_reqs = []
    tr_dv01_fills = []
    tr_alphas = []
    tr_fair_types = []
    tr_vwaps = []
    tr_mids = []
    tr_costs = []
    tr_codes = []
    tr_hedge_sizes_list = []
    tr_hedge_vwaps_list = []
    tr_hedge_fills_list = []
    tr_hedge_costs = []

    total_trades = 0
    for r, t_offset in chunk_results:
        n = r.n_trades
        if n == 0:
            continue
        tr_bars.append(r.tr_bar + t_offset)
        tr_instruments.append(r.tr_instrument)
        tr_sides.append(r.tr_side)
        tr_qty_reqs.append(r.tr_qty_req)
        tr_qty_fills.append(r.tr_qty_fill)
        tr_dv01_reqs.append(r.tr_dv01_req)
        tr_dv01_fills.append(r.tr_dv01_fill)
        tr_alphas.append(r.tr_alpha)
        tr_fair_types.append(r.tr_fair_type)
        tr_vwaps.append(r.tr_vwap)
        tr_mids.append(r.tr_mid)
        tr_costs.append(r.tr_cost)
        tr_codes.append(r.tr_code)
        tr_hedge_sizes_list.append(r.tr_hedge_sizes)
        tr_hedge_vwaps_list.append(r.tr_hedge_vwaps)
        tr_hedge_fills_list.append(r.tr_hedge_fills)
        tr_hedge_costs.append(r.tr_hedge_cost)
        total_trades += n

    def _concat_or_empty(arrays, dtype=np.float64):
        if arrays:
            return np.concatenate(arrays)
        return np.empty(0, dtype=dtype)

    def _concat_2d_or_empty(arrays, n_cols, dtype=np.float64):
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.empty((0, n_cols), dtype=dtype)

    H = first.hedge_positions.shape[1] if first.hedge_positions.ndim == 2 else 1

    # Stitch context arrays
    datetimes = None
    if first.datetimes is not None:
        datetimes = np.concatenate([r.datetimes for r, _ in chunk_results])

    mid_px = None
    if first.mid_px is not None:
        mid_px = np.concatenate([r.mid_px for r, _ in chunk_results], axis=0)

    hedge_mid_px = None
    if first.hedge_mid_px is not None:
        hedge_mid_px = np.concatenate([r.hedge_mid_px for r, _ in chunk_results], axis=0)

    hedge_bid_px = None
    if first.hedge_bid_px is not None:
        hedge_bid_px = np.concatenate([r.hedge_bid_px for r, _ in chunk_results], axis=0)

    hedge_ask_px = None
    if first.hedge_ask_px is not None:
        hedge_ask_px = np.concatenate([r.hedge_ask_px for r, _ in chunk_results], axis=0)

    hedge_ratios = None
    if first.hedge_ratios is not None:
        hedge_ratios = np.concatenate([r.hedge_ratios for r, _ in chunk_results], axis=0)

    dv01_arr = None
    if first.dv01 is not None:
        dv01_arr = np.concatenate([r.dv01 for r, _ in chunk_results], axis=0)

    hedge_dv01 = None
    if first.hedge_dv01 is not None:
        hedge_dv01 = np.concatenate([r.hedge_dv01 for r, _ in chunk_results], axis=0)

    return PortfolioBacktestResults(
        positions=positions,
        cash=cash,
        equity=equity,
        pnl=pnl,
        gross_pnl=gross_pnl,
        codes=codes,
        n_trades_bar=n_trades_bar,
        cooldown=cooldown,
        hedge_positions=hedge_positions,
        hedge_pnl=hedge_pnl,
        instrument_position_mtm=instrument_position_mtm,
        hedge_position_mtm=hedge_position_mtm,
        instrument_cash_mtm=instrument_cash_mtm,
        hedge_cash_mtm=hedge_cash_mtm,
        portfolio_mtm=portfolio_mtm,
        instrument_cost=instrument_cost,
        hedge_cost_bar=hedge_cost_bar,
        portfolio_cost=portfolio_cost,
        tr_bar=_concat_or_empty(tr_bars, np.int64),
        tr_instrument=_concat_or_empty(tr_instruments, np.int32),
        tr_side=_concat_or_empty(tr_sides, np.int32),
        tr_qty_req=_concat_or_empty(tr_qty_reqs),
        tr_qty_fill=_concat_or_empty(tr_qty_fills),
        tr_dv01_req=_concat_or_empty(tr_dv01_reqs),
        tr_dv01_fill=_concat_or_empty(tr_dv01_fills),
        tr_alpha=_concat_or_empty(tr_alphas),
        tr_fair_type=_concat_or_empty(tr_fair_types, np.int32),
        tr_vwap=_concat_or_empty(tr_vwaps),
        tr_mid=_concat_or_empty(tr_mids),
        tr_cost=_concat_or_empty(tr_costs),
        tr_code=_concat_or_empty(tr_codes, np.int32),
        tr_hedge_sizes=_concat_2d_or_empty(tr_hedge_sizes_list, H),
        tr_hedge_vwaps=_concat_2d_or_empty(tr_hedge_vwaps_list, H),
        tr_hedge_fills=_concat_2d_or_empty(tr_hedge_fills_list, H),
        tr_hedge_cost=_concat_or_empty(tr_hedge_costs),
        n_trades=total_trades,
        instrument_ids=first.instrument_ids,
        datetimes=datetimes,
        close_time=first.close_time,
        mid_px=mid_px,
        hedge_mid_px=hedge_mid_px,
        hedge_bid_px=hedge_bid_px,
        hedge_ask_px=hedge_ask_px,
        hedge_ratios=hedge_ratios,
        dv01=dv01_arr,
        hedge_dv01=hedge_dv01,
    )
