#!/usr/bin/env python3
"""
Generate synthetic parquet inputs compatible with BacktestDataLoader.

Creates:
  - book.parquet           (datetime, instrument_id, bid/ask price/size levels)
  - mid.parquet            (datetime, instrument_id, mid_px)
  - dv01.parquet           (datetime, instrument_id, dv01)
  - signal.parquet         (datetime, instrument_id, zscore, expected_yield_pnl_bps, package_yield_bps)
  - hedge_ratios.parquet   (datetime, instrument_id, hedge_ratio)

All are long-format (datetime × instrument_id) to match BacktestDataLoader. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------
# Helpers (reuse from your script)
# ----------------------------
def _make_datetimes(start: str, periods: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=freq, tz=None)


def _random_walk(rng: np.random.Generator, n: int, step_scale: float) -> np.ndarray:
    steps = rng.normal(loc=0.0, scale=step_scale, size=n)
    return np.cumsum(steps)


def _drop_rows(df: pd.DataFrame, rng: np.random.Generator, p: float) -> pd.DataFrame:
    if p <= 0:
        return df
    mask = rng.random(len(df)) >= p
    return df.loc[mask].reset_index(drop=True)


def _make_mid_df(
    rng: np.random.Generator,
    dts: pd.DatetimeIndex,
    ids: list[str],
    jitter_bps: float,
    base0: float,
    base_step: float,
    common_scale: float = 1.0,
    idio_scale: float = 0.6,
    id_col: str = "instrument_id",
) -> pd.DataFrame:
    """
    Make correlated mid prices for a list of instruments.
    mid is roughly in price units around base0 with small random-walk moves.
    """
    T = len(dts)
    # bps -> price units (approx); you used jitter_bps * 1e-4
    common = _random_walk(rng, T, step_scale=(jitter_bps * common_scale) * 1e-4)

    frames = []
    for j, inst in enumerate(ids):
        inst_noise = _random_walk(rng, T, step_scale=(jitter_bps * idio_scale) * 1e-4)
        base = base0 + base_step * j
        mid = (base + common + inst_noise).astype(np.float64)
        frames.append(pd.DataFrame({"datetime": dts, id_col: inst, "mid_px": mid}))
    return pd.concat(frames, ignore_index=True)


def _make_dv01_df(
    rng: np.random.Generator,
    dts: pd.DatetimeIndex,
    ids: list[str],
    base0: float,
    base_step: float,
    noise: float,
    id_col: str,
) -> pd.DataFrame:
    """
    Simple positive DV01 series per instrument (mostly stable).
    """
    T = len(dts)
    frames = []
    for j, inst in enumerate(ids):
        base = base0 + base_step * j
        dv01 = (base + rng.normal(0, noise, size=T)).astype(np.float64)
        dv01 = np.clip(dv01, 1e-6, None)
        frames.append(pd.DataFrame({"datetime": dts, id_col: inst, "dv01": dv01}))
    return pd.concat(frames, ignore_index=True)


def _make_book_df(
    rng: np.random.Generator,
    dts: pd.DatetimeIndex,
    mid_df: pd.DataFrame,
    ids: list[str],
    levels: int,
    spread0: float,
    spread_step: float,
    size0: float,
    size_step: float,
    id_col: str,
    # IMPORTANT: column naming here matches your PortfolioDataLoader expectations:
    # bid_px_0/bid_sz_0/ask_px_0/ask_sz_0 ...
) -> pd.DataFrame:
    T = len(dts)
    mid_pivot = mid_df.pivot(index="datetime", columns=id_col, values="mid_px").reindex(dts)

    frames = []
    for inst in ids:
        inst_mid = mid_pivot[inst].to_numpy(dtype=np.float64)

        # If mid_df had missing rows, pivot introduces NaNs; keep them (gaps are fine),
        # but when computing book, NaNs will propagate. That’s okay if loader ffill/drop.
        rows: dict[str, object] = {"datetime": dts, id_col: inst}

        for l in range(levels):
            spread = (spread0 + spread_step * l)  # in price units
            bid = (inst_mid - spread).astype(np.float64)
            ask = (inst_mid + spread).astype(np.float64)

            # sizes
            bid_sz = (size0 + size_step * l) * (1.0 + rng.normal(0, 0.05, size=T))
            ask_sz = (size0 + size_step * l) * (1.0 + rng.normal(0, 0.05, size=T))

            rows[f"bid_px_l{l}"] = bid
            rows[f"ask_px_l{l}"] = ask
            rows[f"bid_sz_l{l}"] = np.maximum(1.0, bid_sz).astype(np.float64)
            rows[f"ask_sz_l{l}"] = np.maximum(1.0, ask_sz).astype(np.float64)

        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, ignore_index=True)
@dataclass
class GenConfig:
    out_dir: Path
    instrument_ids: list[str]
    ref_instrument_id: str
    start: str
    periods: int
    freq: str
    levels: int
    seed: int
    missing_prob: float  # probability per row to drop (simulate gaps)
    jitter_bps: float    # typical mid move scale in bps per step


def _make_datetimes(start: str, periods: int, freq: str) -> pd.DatetimeIndex:
    # Keep timezone-naive; loader uses pd.DatetimeIndex and .values datetime64. :contentReference[oaicite:2]{index=2}
    return pd.date_range(start=start, periods=periods, freq=freq, tz=None)


def _random_walk(rng: np.random.Generator, n: int, step_scale: float) -> np.ndarray:
    steps = rng.normal(loc=0.0, scale=step_scale, size=n)
    return np.cumsum(steps)


def _drop_rows(df: pd.DataFrame, rng: np.random.Generator, p: float) -> pd.DataFrame:
    if p <= 0:
        return df
    mask = rng.random(len(df)) >= p
    return df.loc[mask].reset_index(drop=True)


def generate_parquets(cfg: GenConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    dts = _make_datetimes(cfg.start, cfg.periods, cfg.freq)
    instruments = cfg.instrument_ids
    if cfg.ref_instrument_id not in instruments:
        raise ValueError(f"ref_instrument_id {cfg.ref_instrument_id!r} not in instrument_ids")

    # ---- Base "mid" generation per instrument ---------------------------------
    # Create smooth mid prices around ~100 with correlated moves.
    T = len(dts)
    common = _random_walk(rng, T, step_scale=cfg.jitter_bps * 1e-4)  # bps -> price units
    mid_frames = []
    for j, inst in enumerate(instruments):
        inst_noise = _random_walk(rng, T, step_scale=(cfg.jitter_bps * 0.6) * 1e-4)
        base = 100.0 + 0.25 * j  # slight level shift per instrument
        mid = base + common + inst_noise
        mid_frames.append(
            pd.DataFrame(
                {
                    "datetime": dts,
                    "instrument_id": inst,
                    "mid_px": mid.astype(np.float64),
                }
            )
        )
    mid_df = pd.concat(mid_frames, ignore_index=True)
    mid_df = _drop_rows(mid_df, rng, cfg.missing_prob)
    mid_df.to_parquet(cfg.out_dir / "mid.parquet", index=False)

    # ---- DV01 per instrument ---------------------------------------------------
    # Simple positive values, mostly stable with small noise.
    dv01_frames = []
    for j, inst in enumerate(instruments):
        base = 0.05 + 0.01 * j
        dv01 = base + rng.normal(0, 0.001, size=T)
        dv01_frames.append(
            pd.DataFrame(
                {
                    "datetime": dts,
                    "instrument_id": inst,
                    "dv01": dv01.astype(np.float64),
                }
            )
        )
    dv01_df = pd.concat(dv01_frames, ignore_index=True)
    dv01_df = _drop_rows(dv01_df, rng, cfg.missing_prob)
    dv01_df.to_parquet(cfg.out_dir / "dv01.parquet", index=False)

    # ---- Book levels (bid/ask px/sz) ------------------------------------------
    book_df = _make_book_df(
        rng, dts, mid_df, cfg.instrument_ids, cfg.levels,
        spread0=0.008, spread_step=0.003,  # similar to your single-instrument generator
        size0=100.0, size_step=50.0,
        id_col="instrument_id",
    )
    book_df = _drop_rows(book_df, rng, cfg.missing_prob)
    book_df.to_parquet(cfg.out_dir / "book.parquet", index=False)

    # ---- Signal (for ALL instruments) --------------------------------------------

    signal_frames = []

    # horizon in bars for "1 day ahead" forecast.
    # If your bar is 5min, 1 day ≈ 288; if unknown, pick something stable.
    horizon = getattr(cfg, "fair_horizon_bars", None)
    if horizon is None:
        horizon = min(288, max(1, T // 5))

    for j, inst in enumerate(instruments):
        # --- z-score AR-ish process ---
        z = rng.normal(0, 1.0, size=T)
        for t in range(1, T):
            z[t] = 0.92 * z[t - 1] + 0.35 * z[t]

        # Make each instrument slightly different
        z = z + 0.15 * j

        expected_yield_pnl_bps = (-0.8 * z + rng.normal(0, 0.2, size=T)).astype(np.float64)
        package_yield_bps = (0.5 * z + rng.normal(0, 0.2, size=T)).astype(np.float64)

        # --- NEW: simulate a mid price and a 1-day ahead fair_price forecast ---
        # mid price random walk around a level; make instruments differ slightly
        mid0 = 100.0 + 2.0 * j + rng.normal(0, 1.0)
        mid = np.empty(T, dtype=np.float64)
        mid[0] = mid0
        for t in range(1, T):
            mid[t] = mid[t - 1] + rng.normal(0, 0.03)  # small drift/vol

        # create fair = future mid (shifted) + forecast noise
        fair_price = np.empty(T, dtype=np.float64)
        if horizon < T:
            fair_price[:-horizon] = mid[horizon:]
            fair_price[-horizon:] = mid[-1]  # last part: hold flat
        else:
            fair_price[:] = mid[-1]

        # Add model noise (prediction error)
        fair_price += rng.normal(0, 0.05, size=T)

        # --- NEW: adf p-value regime simulation ---
        # 70% mean-reverting regime (small p), 30% non-stationary (large p)
        stable = rng.random(T) < 0.7
        adf_p_value = np.where(
            stable,
            rng.uniform(0.001, 0.08, size=T),  # "passes" typical p-threshold
            rng.uniform(0.12, 0.90, size=T),  # "fails" ADF gate
        ).astype(np.float64)

        signal_frames.append(
            pd.DataFrame(
                {
                    "datetime": dts,
                    "instrument_id": inst,
                    "zscore": z.astype(np.float64),
                    "expected_yield_pnl_bps": expected_yield_pnl_bps,
                    "package_yield_bps": package_yield_bps,
                    # NEW columns
                    "fair_price": fair_price,
                    "adf_p_value": adf_p_value,
                }
            )
        )

    signal_df = pd.concat(signal_frames, ignore_index=True)
    signal_df = _drop_rows(signal_df, rng, cfg.missing_prob)
    signal_df.to_parquet(cfg.out_dir / "signal.parquet", index=False)

    # ---- Hedge ratios (daily, list schema) -------------------------------------
    # Hedge is daily (or business-daily) and must be forward-filled when aligning to intraday dts.

    # Build daily timestamps spanning the intraday range
    start_dt = pd.Timestamp(dts[0]).normalize()
    end_dt = pd.Timestamp(dts[-1]).normalize()

    # Use "D" for calendar daily, or "B" for business day
    hedge_dts = pd.date_range(start=start_dt, end=end_dt, freq="B")

    phi = 0.98  # smooth day-to-day changes
    eps_scale = 0.15  # how much it changes each day

    state = {target: None for target in instruments}

    hedge_rows = []
    for dt_idx, dt in enumerate(hedge_dts):
        for target in instruments:
            if dt_idx < len(hedge_dts)/2:
                hedge_legs = [x for x in instruments if x != target][:2]
            else:
                hedge_legs = [x for x in instruments if x != target][1:]
            M = len(hedge_legs)

            if state[target] is None:
                state[target] = rng.normal(0, 1.0, size=M)

            # AR(1) daily update
            state[target] = phi * state[target] + rng.normal(0.0, eps_scale, size=M)

            # Convert to weights that sum to 1
            x = state[target] - np.max(state[target])
            w = np.exp(x)
            w = w / (np.sum(w) + 1e-12)

            # Make hedge ratios negative and sum to -1
            ratios = (-w).astype(np.float64)  # sum = -1

            hedge_rows.append(
                {
                    "datetime": dt,
                    "instrument_id": target,  # TARGET being hedged
                    "hedge_instruments": hedge_legs,  # excludes target
                    "hedge_ratios": [float(r) for r in ratios],
                }
            )

    hedge_df = pd.DataFrame(hedge_rows)
    hedge_df = _drop_rows(hedge_df, rng, cfg.missing_prob)
    hedge_df.to_parquet(cfg.out_dir / "hedge_ratios.parquet", index=False)

    print(f"Wrote synthetic parquets to: {cfg.out_dir.resolve()}")
    print("Files:", ", ".join(sorted(p.name for p in cfg.out_dir.glob("*.parquet"))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory, e.g. data/sim_20260217")
    ap.add_argument("--instruments", type=str, default="UST_2Y,UST_5Y,UST_10Y,UST_30Y",
                    help="Comma-separated instrument_ids in desired order")
    ap.add_argument("--ref", type=str, default="UST_5Y", help="ref_instrument_id for signal")
    ap.add_argument("--start", type=str, default="2026-01-01 09:00:00", help="Start datetime (naive)")
    ap.add_argument("--periods", type=int, default=20000, help="Number of timestamps")
    ap.add_argument("--freq", type=str, default="5min", help="Pandas date_range freq, e.g. 1min, 5s, 1H")
    ap.add_argument("--levels", type=int, default=5, help="Number of book levels L")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed")
    ap.add_argument("--missing-prob", type=float, default=0.02,
                    help="Probability to drop individual rows to create gaps (0..0.3 typical)")
    ap.add_argument("--jitter-bps", type=float, default=1.2,
                    help="Typical mid move scale in bps per step (rough)")
    args = ap.parse_args()

    cfg = GenConfig(
        out_dir=Path(args.out),
        instrument_ids=[s.strip() for s in args.instruments.split(",") if s.strip()],
        ref_instrument_id=args.ref.strip(),
        start=args.start,
        periods=args.periods,
        freq=args.freq,
        levels=args.levels,
        seed=args.seed,
        missing_prob=float(args.missing_prob),
        jitter_bps=float(args.jitter_bps),
    )
    generate_parquets(cfg)


if __name__ == "__main__":
    main()
