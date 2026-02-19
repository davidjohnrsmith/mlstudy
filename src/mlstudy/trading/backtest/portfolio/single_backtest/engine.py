"""High-level entry point for the LP portfolio backtester.

Validates inputs, unpacks the config, calls the loop, and wraps raw
arrays into :class:`PortfolioBacktestResults`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mlstudy.trading.backtest.common.single_backtest.engine import ensure_f64, validate_l2_shapes
from mlstudy.trading.backtest.portfolio.configs.backtest_config import PortfolioBacktestConfig
from mlstudy.trading.backtest.portfolio.single_backtest.loop import lp_portfolio_loop
from mlstudy.trading.backtest.portfolio.single_backtest.results import PortfolioBacktestResults


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
    hedge_bid_px: Optional[np.ndarray],
    hedge_bid_sz: Optional[np.ndarray],
    hedge_ask_px: Optional[np.ndarray],
    hedge_ask_sz: Optional[np.ndarray],
    hedge_mid_px: Optional[np.ndarray],
    hedge_dv01: Optional[np.ndarray],
    hedge_ratios: Optional[np.ndarray],
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
    ]:
        if len(arr) != expected_len:
            raise ValueError(f"{name} length {len(arr)} != expected {expected_len}")

    # Hedge arrays: all-or-nothing
    hedge_arrays = [
        hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz,
        hedge_mid_px, hedge_dv01, hedge_ratios,
    ]
    n_provided = sum(a is not None for a in hedge_arrays)
    if n_provided not in (0, 7):
        raise ValueError(
            f"Hedge arrays must be all-None or all-provided, got {n_provided}/7"
        )

    if n_provided == 7:
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
    # -- Optional meta --
    maturity: Optional[np.ndarray] = None,
    issuer_bucket: Optional[np.ndarray] = None,
    maturity_bucket: Optional[np.ndarray] = None,
    # -- Optional bucket caps --
    issuer_dv01_caps: Optional[np.ndarray] = None,
    mat_bucket_dv01_caps: Optional[np.ndarray] = None,
    # -- Hedge arrays (all-or-nothing) --
    hedge_bid_px: Optional[np.ndarray] = None,
    hedge_bid_sz: Optional[np.ndarray] = None,
    hedge_ask_px: Optional[np.ndarray] = None,
    hedge_ask_sz: Optional[np.ndarray] = None,
    hedge_mid_px: Optional[np.ndarray] = None,
    hedge_dv01: Optional[np.ndarray] = None,
    hedge_ratios: Optional[np.ndarray] = None,
    # -- Config --
    cfg: Optional[PortfolioBacktestConfig] = None,
    # -- Optional context --
    datetimes: Optional[np.ndarray] = None,
) -> PortfolioBacktestResults:
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
    maturity : (B,) or None
        Years to maturity per instrument.
    issuer_bucket, maturity_bucket : (B,) or None
        Bucket labels per instrument.
    issuer_dv01_caps : (n_issuers,) or None
    mat_bucket_dv01_caps : (n_buckets,) or None
    hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz : (T, H, L) or None
        L2 order book for hedge instruments.
    hedge_mid_px : (T, H) or None
    hedge_dv01 : (T, H) or None
    hedge_ratios : (T, B, H) or None
    cfg : PortfolioBacktestConfig, optional
        Scalar parameters.  Defaults are used if *None*.
    datetimes : (T,) or None
        Bar timestamps for DataFrame output.

    Returns
    -------
    PortfolioBacktestResults
    """
    if cfg is None:
        cfg = PortfolioBacktestConfig()

    _validate(
        bid_px, bid_sz, ask_px, ask_sz, mid_px,
        dv01, fair_price, zscore, adf_p_value,
        tradable, pos_limits_long, pos_limits_short,
        hedge_bid_px, hedge_bid_sz, hedge_ask_px, hedge_ask_sz,
        hedge_mid_px, hedge_dv01, hedge_ratios,
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

    # Hedge arrays
    h_kwargs = {}
    if hedge_bid_px is not None:
        h_kwargs = dict(
            hedge_bid_px=ensure_f64(hedge_bid_px),
            hedge_bid_sz=ensure_f64(hedge_bid_sz),
            hedge_ask_px=ensure_f64(hedge_ask_px),
            hedge_ask_sz=ensure_f64(hedge_ask_sz),
            hedge_mid_px=ensure_f64(hedge_mid_px),
            hedge_dv01=ensure_f64(hedge_dv01),
            hedge_ratios=ensure_f64(hedge_ratios),
        )

    raw = lp_portfolio_loop(
        f_bid_px, f_bid_sz, f_ask_px, f_ask_sz, f_mid_px,
        f_dv01, f_fair, f_zscore, f_adf,
        tradable, pos_limits_long, pos_limits_short,
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
        qty_step=float(cfg.qty_step),
        min_qty_trade=float(cfg.min_qty_trade),
        min_fill_ratio=float(cfg.min_fill_ratio),
        cooldown_bars=int(cfg.cooldown_bars),
        cooldown_mode=int(cfg.cooldown_mode),
        min_maturity_inc=float(cfg.min_maturity_inc),
        initial_capital=float(cfg.initial_capital),
        **h_kwargs,
    )

    return PortfolioBacktestResults.from_loop_output(
        raw,
        datetimes=datetimes,
        mid_px=f_mid_px,
        hedge_mid_px=h_kwargs.get("hedge_mid_px"),
    )
