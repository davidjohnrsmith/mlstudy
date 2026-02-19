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


def _make_meta_df(
    rng: np.random.Generator,
    instrument_ids: list[str],
) -> pd.DataFrame:
    B = len(instrument_ids)
    tradable = (rng.random(B) > 0.05).astype(float)
    pos_limit_long = rng.uniform(1e6, 5e6, size=B)
    pos_limit_short = -rng.uniform(1e6, 5e6, size=B)

    maturity = rng.uniform(0.5, 12.0, size=B)
    issuer_bucket = rng.integers(0, 10, size=B)
    maturity_bucket = np.digitize(maturity, bins=[2, 5, 7, 10]).astype(int)

    return pd.DataFrame({
        "instrument_id": instrument_ids,
        "tradable": tradable,
        "pos_limit_long": pos_limit_long,
        "pos_limit_short": pos_limit_short,
        "maturity": maturity,
        "issuer_bucket": issuer_bucket,
        "maturity_bucket": maturity_bucket,
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

    # ---- instrument mid / dv01 / book ----
    instrument_mid_df = _make_mid_df(
        rng, dts, cfg.instrument_ids, cfg.jitter_bps,
        base0=100.0, base_step=0.25,
        id_col="instrument_id",
    )
    instrument_mid_df = _drop_rows(instrument_mid_df, rng, cfg.missing_prob)
    instrument_mid_df.to_parquet(cfg.out_dir / "mid.parquet", index=False)

    instrument_dv01_df = _make_dv01_df(
        rng, dts, cfg.instrument_ids,
        base0=0.05, base_step=0.01, noise=0.001,
        id_col="instrument_id",
    )
    instrument_dv01_df = _drop_rows(instrument_dv01_df, rng, cfg.missing_prob)
    instrument_dv01_df.to_parquet(cfg.out_dir / "dv01.parquet", index=False)

    instrument_book_df = _make_book_df(
        rng, dts, instrument_mid_df, cfg.instrument_ids, cfg.instrument_levels,
        spread0=0.008, spread_step=0.003,   # similar to your single-instrument generator
        size0=100.0, size_step=50.0,
        id_col="instrument_id",
    )
    instrument_book_df = _drop_rows(instrument_book_df, rng, cfg.missing_prob)
    instrument_book_df.to_parquet(cfg.out_dir / "book.parquet", index=False)

    # ---- Hedge mid / dv01 / book ----
    hedge_mid_df = _make_mid_df(
        rng, dts, cfg.hedge_ids, cfg.jitter_bps * 0.8,
        base0=100.0, base_step=0.10,
        id_col="hedge_id",
    )
    hedge_mid_df = _drop_rows(hedge_mid_df, rng, cfg.missing_prob)
    hedge_mid_df.to_parquet(cfg.out_dir / "hedge_mid.parquet", index=False)

    hedge_dv01_df = _make_dv01_df(
        rng, dts, cfg.hedge_ids,
        base0=0.06, base_step=0.008, noise=0.001,
        id_col="hedge_id",
    )
    hedge_dv01_df = _drop_rows(hedge_dv01_df, rng, cfg.missing_prob)
    hedge_dv01_df.to_parquet(cfg.out_dir / "hedge_dv01.parquet", index=False)

    hedge_book_df = _make_book_df(
        rng, dts, hedge_mid_df, cfg.hedge_ids, cfg.hedge_levels,
        spread0=0.006, spread_step=0.002,   # a bit tighter for hedges
        size0=150.0, size_step=60.0,
        id_col="hedge_id",
    )
    hedge_book_df = _drop_rows(hedge_book_df, rng, cfg.missing_prob)
    hedge_book_df.to_parquet(cfg.out_dir / "hedge_book.parquet", index=False)

    # ---- Signals (for instruments only) ----
    # IMPORTANT: derived from instrument_mid_df so consistent with book/mid
    signal_df = _make_signal_df_for_instruments(
        rng, dts, cfg.instrument_ids, instrument_mid_df,
        missing_prob=cfg.missing_prob,
        fair_horizon_bars=cfg.fair_horizon_bars,
    )
    signal_df.to_parquet(cfg.out_dir / "signal.parquet", index=False)

    # ---- Meta (instruments) ----
    meta_df = _make_meta_df(rng, cfg.instrument_ids)
    meta_df.to_parquet(cfg.out_dir / "meta.parquet", index=False)

    # ---- Hedge ratios list parquet ----
    hedge_ratios_df = _make_hedge_ratios_list_parquet(
        rng, dts, cfg.instrument_ids, cfg.hedge_ids, missing_prob=cfg.missing_prob,
    )
    hedge_ratios_df.to_parquet(cfg.out_dir / "hedge_ratios.parquet", index=False)

    print(f"Wrote synthetic portfolio parquets to: {cfg.out_dir.resolve()}")
    print("Files:", ", ".join(sorted(p.name for p in cfg.out_dir.glob("*.parquet"))))


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
    generate_portfolio_parquets(cfg)


if __name__ == "__main__":
    main()
