#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlstudy.core.data.simulation.backtest import _drop_rows, _make_datetimes, _make_mid_df, _make_dv01_df, \
    _make_book_df


# ----------------------------
# Config
# ----------------------------
@dataclass
class PortfolioGenConfig:
    out_dir: Path
    instrument_ids: list[str]
    hedge_ids: list[str]
    start: str
    periods: int
    freq: str
    instrument_levels: int
    hedge_levels: int
    seed: int
    missing_prob: float
    jitter_bps: float
    fair_horizon_bars: int | None = None





def _make_signal_df_for_instruments(
    rng: np.random.Generator,
    dts: pd.DatetimeIndex,
    instrument_ids: list[str],
    instrument_mid_df: pd.DataFrame,
    missing_prob: float,
    fair_horizon_bars: int | None,
) -> pd.DataFrame:
    """
    Signal parquet for portfolio loader:
      datetime, instrument_id, fair_price, zscore, adf_p_value
    fair/zscore are derived from instrument mid to be consistent.
    """
    T = len(dts)
    mid_pivot = instrument_mid_df.pivot(index="datetime", columns="instrument_id", values="mid_px").reindex(dts)

    horizon = fair_horizon_bars
    if horizon is None:
        horizon = min(288, max(1, T // 5))

    frames = []
    for j, b in enumerate(instrument_ids):
        mid = mid_pivot[b].to_numpy(dtype=np.float64)
        if np.isnan(mid).any():
            mid = pd.Series(mid, index=dts).ffill().bfill().to_numpy(dtype=np.float64)

        # fair = forward-shifted mid + noise
        fair = np.empty(T, dtype=np.float64)
        if horizon < T:
            fair[:-horizon] = mid[horizon:]
            fair[-horizon:] = mid[-1]
        else:
            fair[:] = mid[-1]
        fair += rng.normal(0, 0.05, size=T)

        # zscore correlated with (fair - mid) + AR-ish component
        spread_like = fair - mid
        denom = np.std(spread_like) + 1e-8
        z_corr = (spread_like / denom) * 1.2 + rng.normal(0, 0.2, size=T)

        z_ar = rng.normal(0, 1.0, size=T)
        for t in range(1, T):
            z_ar[t] = 0.92 * z_ar[t - 1] + 0.35 * z_ar[t]
        z_ar = z_ar + 0.15 * j

        zscore = (0.6 * z_ar + 0.4 * z_corr).astype(np.float64)

        stable = rng.random(T) < 0.7
        adf_p = np.where(
            stable,
            rng.uniform(0.001, 0.08, size=T),
            rng.uniform(0.12, 0.90, size=T),
        ).astype(np.float64)

        frames.append(pd.DataFrame({
            "datetime": dts,
            "instrument_id": b,
            "fair_price": fair.astype(np.float64),
            "zscore": zscore,
            "adf_p_value": adf_p,
        }))

    signal_df = pd.concat(frames, ignore_index=True)
    signal_df = _drop_rows(signal_df, rng, missing_prob)
    return signal_df


_ISSUER_NAMES = ["US",]


def _make_meta_df(
    rng: np.random.Generator,
    instrument_ids: list[str],
    ref_date: str = "2026-01-01",
) -> pd.DataFrame:
    B = len(instrument_ids)
    tradable = (rng.random(B) > 0.05).astype(float)
    pos_limit_long = rng.uniform(1e6, 5e6, size=B)
    pos_limit_short = -rng.uniform(1e6, 5e6, size=B)

    # maturity_date: random date 0.5–12 years from ref_date
    ref = pd.Timestamp(ref_date)
    days_ahead = rng.uniform(0.5 * 365.25, 12.0 * 365.25, size=B).astype(int)
    maturity_date = pd.to_datetime(
        [ref + pd.Timedelta(days=int(d)) for d in days_ahead]
    )

    # issuer: string names (loader maps these via issuer_dv01_caps_map)
    issuer_idx = rng.integers(0, len(_ISSUER_NAMES), size=B)
    issuer = [_ISSUER_NAMES[i] for i in issuer_idx]

    max_trade_notional_inc = rng.uniform(5e5, 2e6, size=B)
    max_trade_notional_dec = rng.uniform(5e5, 2e6, size=B)
    qty_step = rng.choice([1000.0, 5000.0, 10000.0], size=B)

    return pd.DataFrame({
        "instrument_id": instrument_ids,
        "tradable": tradable,
        "pos_limit_long": pos_limit_long,
        "pos_limit_short": pos_limit_short,
        "max_trade_notional_inc": max_trade_notional_inc,
        "max_trade_notional_dec": max_trade_notional_dec,
        "qty_step": qty_step,
        "maturity_date": maturity_date,
        "issuer_bucket": issuer,
    })


def _make_hedge_meta_df(
    rng: np.random.Generator,
    hedge_ids: list[str],
) -> pd.DataFrame:
    """Create hedge meta DataFrame with per-hedge qty_step."""
    H = len(hedge_ids)
    qty_step = rng.choice([1000.0, 5000.0, 10000.0], size=H)
    return pd.DataFrame({
        "instrument_id": hedge_ids,
        "qty_step": qty_step,
    })


def _make_hedge_ratios_list_parquet(
    rng: np.random.Generator,
    dts: pd.DatetimeIndex,
    instrument_ids: list[str],
    hedge_ids: list[str],
    missing_prob: float,
) -> pd.DataFrame:
    """
    List-column format required by your PortfolioDataLoader:
      datetime, instrument_id, hedge_instruments(list[str]), hedge_ratios(list[float])
    Ratios sum to -1 (implicit main weight +1 => sum=0).
    """
    H = len(hedge_ids)
    rows = []
    for dt in dts:
        for b in instrument_ids:
            k = int(rng.integers(1, min(H, 3) + 1))
            active = rng.choice(H, size=k, replace=False)

            w = rng.normal(0, 1, size=k) - 0.3
            if np.all(np.abs(w) < 1e-8):
                w[0] = -1.0
            w = w / np.sum(w) * (-1.0)  # sum to -1

            rows.append({
                "datetime": dt,
                "instrument_id": b,
                "hedge_instruments": [hedge_ids[h] for h in active],
                "hedge_ratios": [float(x) for x in w],
            })
    df = pd.DataFrame(rows)
    df = _drop_rows(df, rng, missing_prob)
    return df


# ----------------------------
# Main generator (portfolio parquets)
# ----------------------------
def generate_portfolio_parquets(cfg: PortfolioGenConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    dts = _make_datetimes(cfg.start, cfg.periods, cfg.freq)

    instrument_ids = list(cfg.instrument_ids)
    hedge_ids = list(cfg.hedge_ids)
    all_ids = instrument_ids + hedge_ids

    # ---- mid / dv01 / book for ALL ids (instruments + hedges) -----------------
    mid_df = _make_mid_df(
        rng,
        dts,
        all_ids,
        cfg.jitter_bps,
        base0=100.0,
        base_step=0.20,
        id_col="instrument_id",
    )
    mid_df = _drop_rows(mid_df, rng, cfg.missing_prob)
    mid_df.to_parquet(cfg.out_dir / "mid.parquet", index=False)

    dv01_df = _make_dv01_df(
        rng,
        dts,
        all_ids,
        base0=0.05,
        base_step=0.01,
        noise=0.001,
        id_col="instrument_id",
    )
    dv01_df = _drop_rows(dv01_df, rng, cfg.missing_prob)
    dv01_df.to_parquet(cfg.out_dir / "dv01.parquet", index=False)

    book_df = _make_book_df(
        rng,
        dts,
        mid_df,
        all_ids,
        cfg.instrument_levels,      # one L for everything (loader detects from book columns)
        spread0=0.008,
        spread_step=0.003,
        size0=120.0,
        size_step=60.0,
        id_col="instrument_id",
    )
    book_df = _drop_rows(book_df, rng, cfg.missing_prob)
    book_df.to_parquet(cfg.out_dir / "book.parquet", index=False)

    # ---- Signals (for instruments only) --------------------------------------
    # Must contain: fair_price, zscore, adf_p_value, and be long-format on (datetime, instrument_id)
    signal_df = _make_signal_df_for_instruments(
        rng,
        dts,
        instrument_ids,
        mid_df,                     # use mid_df so signals are aligned with mid
        missing_prob=cfg.missing_prob,
        fair_horizon_bars=cfg.fair_horizon_bars,
    )
    signal_df.to_parquet(cfg.out_dir / "signal.parquet", index=False)

    # ---- Meta (instruments only) ---------------------------------------------
    meta_df = _make_meta_df(
        rng,
        instrument_ids,
        ref_date=cfg.start,
    )
    meta_df.to_parquet(cfg.out_dir / "meta.parquet", index=False)

    # ---- Hedge ratios list parquet (targets are instruments; hedges are hedge_ids) ----
    hedge_ratios_df = _make_hedge_ratios_list_parquet(
        rng,
        dts,
        instrument_ids,
        hedge_ids,
        missing_prob=cfg.missing_prob,
    )
    hedge_ratios_df.to_parquet(cfg.out_dir / "hedge_ratios.parquet", index=False)

    # ---- Hedge meta (hedge instruments only) ---------------------------------
    hedge_meta_df = _make_hedge_meta_df(rng, hedge_ids)
    hedge_meta_df.to_parquet(cfg.out_dir / "hedge_meta.parquet", index=False)

    print(f"Wrote synthetic portfolio parquets to: {cfg.out_dir.resolve()}")
    print("Files:", ", ".join(sorted(p.name for p in cfg.out_dir.glob("*.parquet"))))


def _split_df_by_date(
    df: pd.DataFrame,
    dt_col: str,
    chunk_boundaries: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[str, str, pd.DataFrame]]:
    """Split a DataFrame into chunks by date boundaries.

    Returns list of (start_date_str, end_date_str, chunk_df).
    """
    dt_series = pd.to_datetime(df[dt_col])
    result = []
    for cs, ce in chunk_boundaries:
        mask = (dt_series >= cs) & (dt_series <= ce)
        chunk = df.loc[mask].reset_index(drop=True)
        if chunk.empty:
            continue
        cs_str = cs.strftime("%Y%m%d")
        ce_str = ce.strftime("%Y%m%d")
        result.append((cs_str, ce_str, chunk))
    return result


def generate_portfolio_parquets_chunked(
    cfg: PortfolioGenConfig,
    chunk_freq: str = "D",
) -> None:
    """Generate date-suffixed chunked parquet files.

    Produces files like ``book_20240101_20240101.parquet`` instead of
    ``book.parquet``.  The ``meta.parquet`` file remains a single file
    (static metadata doesn't change over time).

    Parameters
    ----------
    cfg : PortfolioGenConfig
        Same config as :func:`generate_portfolio_parquets`.
    chunk_freq : str
        Pandas frequency string for chunk boundaries (e.g. ``"D"`` for daily,
        ``"W"`` for weekly, ``"MS"`` for month-start).
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    dts = _make_datetimes(cfg.start, cfg.periods, cfg.freq)

    instrument_ids = list(cfg.instrument_ids)
    hedge_ids = list(cfg.hedge_ids)
    all_ids = instrument_ids + hedge_ids

    # ---- Generate full DataFrames (same logic as single-file) ----------------
    mid_df = _make_mid_df(
        rng, dts, all_ids, cfg.jitter_bps,
        base0=100.0, base_step=0.20, id_col="instrument_id",
    )
    mid_df = _drop_rows(mid_df, rng, cfg.missing_prob)

    dv01_df = _make_dv01_df(
        rng, dts, all_ids,
        base0=0.05, base_step=0.01, noise=0.001, id_col="instrument_id",
    )
    dv01_df = _drop_rows(dv01_df, rng, cfg.missing_prob)

    book_df = _make_book_df(
        rng, dts, mid_df, all_ids, cfg.instrument_levels,
        spread0=0.008, spread_step=0.003,
        size0=120.0, size_step=60.0, id_col="instrument_id",
    )
    book_df = _drop_rows(book_df, rng, cfg.missing_prob)

    signal_df = _make_signal_df_for_instruments(
        rng, dts, instrument_ids, mid_df,
        missing_prob=cfg.missing_prob,
        fair_horizon_bars=cfg.fair_horizon_bars,
    )

    meta_df = _make_meta_df(rng, instrument_ids, ref_date=cfg.start)

    hedge_ratios_df = _make_hedge_ratios_list_parquet(
        rng, dts, instrument_ids, hedge_ids,
        missing_prob=cfg.missing_prob,
    )

    hedge_meta_df = _make_hedge_meta_df(rng, hedge_ids)

    # ---- Compute chunk boundaries -------------------------------------------
    start_dt = dts[0].normalize()
    end_dt = dts[-1].normalize()
    chunk_starts = pd.date_range(start_dt, end_dt, freq=chunk_freq)
    if len(chunk_starts) == 0:
        chunk_starts = pd.DatetimeIndex([start_dt])

    chunk_boundaries = []
    for i, cs in enumerate(chunk_starts):
        if i + 1 < len(chunk_starts):
            ce = chunk_starts[i + 1] - pd.Timedelta(nanoseconds=1)
        else:
            ce = end_dt + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        chunk_boundaries.append((cs, ce))

    # ---- Split and write date-suffixed files --------------------------------
    dt_col = "datetime"
    data_map = {
        "book": book_df,
        "mid": mid_df,
        "dv01": dv01_df,
        "signal": signal_df,
        "hedge_ratios": hedge_ratios_df,
    }

    files_written = []
    for base_name, df in data_map.items():
        chunks = _split_df_by_date(df, dt_col, chunk_boundaries)
        for cs_str, ce_str, chunk_df in chunks:
            fname = f"{base_name}_{cs_str}_{ce_str}.parquet"
            chunk_df.to_parquet(cfg.out_dir / fname, index=False)
            files_written.append(fname)

    # ---- Meta is static — single file ---------------------------------------
    meta_df.to_parquet(cfg.out_dir / "meta.parquet", index=False)
    files_written.append("meta.parquet")

    # ---- Hedge meta is static — single file ---------------------------------
    hedge_meta_df.to_parquet(cfg.out_dir / "hedge_meta.parquet", index=False)
    files_written.append("hedge_meta.parquet")

    print(f"Wrote {len(files_written)} chunked parquet files to: {cfg.out_dir.resolve()}")
    print(f"Chunk freq: {chunk_freq}, chunks: {len(chunk_boundaries)}")
    print("Sample files:", ", ".join(sorted(files_written)[:6]), "...")


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Generate synthetic parquets for PortfolioDataLoader")
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--instruments", type=str, default="instrument_000,instrument_001,instrument_002,instrument_003,instrument_004")
    ap.add_argument("--hedges", type=str, default="HEDGE_00,HEDGE_01,HEDGE_02")
    ap.add_argument("--start", type=str, default="2026-01-01 09:00:00")
    ap.add_argument("--periods", type=int, default=1000)
    ap.add_argument("--freq", type=str, default="1min")
    ap.add_argument("--instrument-levels", type=int, default=5)
    ap.add_argument("--hedge-levels", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--missing-prob", type=float, default=0.02)
    ap.add_argument("--jitter-bps", type=float, default=1.2)
    ap.add_argument("--fair-horizon-bars", type=int, default=None)
    ap.add_argument("--chunked", action="store_true",
                    help="Generate date-suffixed chunked files instead of single files")
    ap.add_argument("--chunk-freq", type=str, default="D",
                    help="Chunk frequency (D=daily, W=weekly, MS=month-start)")
    args = ap.parse_args()

    cfg = PortfolioGenConfig(
        out_dir=Path(args.out),
        instrument_ids=[s.strip() for s in args.instruments.split(",") if s.strip()],
        hedge_ids=[s.strip() for s in args.hedges.split(",") if s.strip()],
        start=args.start,
        periods=args.periods,
        freq=args.freq,
        instrument_levels=args.instrument_levels,
        hedge_levels=args.hedge_levels,
        seed=args.seed,
        missing_prob=float(args.missing_prob),
        jitter_bps=float(args.jitter_bps),
        fair_horizon_bars=args.fair_horizon_bars,
    )
    if args.chunked:
        generate_portfolio_parquets_chunked(cfg, chunk_freq=args.chunk_freq)
    else:
        generate_portfolio_parquets(cfg)


if __name__ == "__main__":
    main()
