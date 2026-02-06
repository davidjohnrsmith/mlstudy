from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, Sequence, Optional, Dict, List, Tuple, Protocol
import pandas as pd
import numpy as np


# -------------------------
# 1) Enum: your 9 types
# -------------------------
class StructureType(Enum):
    BM_FLY_ACROSS_TENORS = auto()          # (1) 2*mid - short - long (virtual benchmark across tenors)
    BM_SWITCH_ACROSS_TENORS = auto()       # (2) long - short (virtual benchmark across tenors)

    BM_FLY_WITHIN_TENOR = auto()           # (3) OTR/prev/preprev within same tenor (concrete bonds)
    BM_SWITCH_WITHIN_TENOR = auto()        # (4) OTR - prev within same tenor (concrete bonds)

    NEARBY_FLY_ACROSS_CURVE = auto()       # (5) any 3 adjacent bonds by TTM
    NEARBY_SWITCH_ACROSS_CURVE = auto()    # (6) any 2 adjacent bonds by TTM

    CTD_FLY = auto()                       # (7) CTD across futures buckets (virtual CTD)
    CTD_SWITCH = auto()                    # (8) CTD switch across futures buckets (virtual CTD)

    CROSS_COUNTRY_SPREAD = auto()          # (9) e.g. BM(DE,10Y) - BM(FR,10Y) (virtual benchmark)


# -------------------------
# 2) Data structures
# -------------------------
@dataclass(frozen=True)
class Leg:
    isin: str
    role: str  # e.g. "short", "mid", "long" or "left"/"right"

@dataclass(frozen=True)
class Structure:
    structure_type: StructureType
    name: str
    country: Optional[str]                 # single-country structures
    countries: Optional[Tuple[str, str]]   # for cross-country spread
    tenor: Optional[str]                   # when relevant (e.g. "10Y")
    tenors: Optional[Tuple[str, ...]]      # when multiple tenors used (e.g. ("2Y","5Y","10Y"))
    legs: Tuple[Leg, ...]                  # concrete ISIN legs
    meta: Dict[str, str]                   # free-form metadata


# -------------------------
# 3) Helpers: resolve mappings as-of a date
# -------------------------
def _as_date(d) -> pd.Timestamp:
    return pd.Timestamp(d).normalize()

def resolve_effective_mapping(
    mapping_df: pd.DataFrame,
    asof: pd.Timestamp,
    keys: Dict[str, str],
    isin_col: str = "isin",
    eff_from_col: str = "effective_from",
    eff_to_col: str = "effective_to",
) -> Optional[str]:
    """
    Resolve a single mapping row as-of date for given keys.
    mapping_df must include columns in keys + effective_from/effective_to + isin.

    effective_to can be NaT/None meaning open-ended.
    Interval: [effective_from, effective_to)  (effective_to exclusive)
    """
    asof = _as_date(asof)

    df = mapping_df
    for k, v in keys.items():
        df = df[df[k] == v]

    if df.empty:
        return None

    ef = pd.to_datetime(df[eff_from_col]).dt.normalize()
    et = pd.to_datetime(df[eff_to_col]).dt.normalize() if eff_to_col in df.columns else pd.Series([pd.NaT]*len(df), index=df.index)

    # open-ended effective_to => treat as far future
    et_filled = et.fillna(pd.Timestamp("2099-12-31"))

    mask = (ef <= asof) & (asof < et_filled)
    df = df.loc[mask]
    if df.empty:
        return None

    # if multiple match, take the one with latest effective_from
    df = df.assign(_ef=ef.loc[df.index]).sort_values("_ef", ascending=False)
    return str(df.iloc[0][isin_col])


def resolve_mapping_history(
    mapping_df: pd.DataFrame,
    asof: pd.Timestamp,
    keys: Dict[str, str],
    isin_col: str = "isin",
    eff_from_col: str = "effective_from",
    eff_to_col: str = "effective_to",
) -> List[str]:
    """
    Return mapping ISIN history ordered from most recent to older, *up to asof*,
    using effective_from ordering. Useful for within-tenor OTR/prev/preprev.
    """
    asof = _as_date(asof)

    df = mapping_df
    for k, v in keys.items():
        df = df[df[k] == v]
    if df.empty:
        return []

    df = df.copy()
    df["_ef"] = pd.to_datetime(df[eff_from_col]).dt.normalize()
    # keep rows that started on/before asof
    df = df[df["_ef"] <= asof]
    if df.empty:
        return []
    df = df.sort_values("_ef", ascending=False)
    return [str(x) for x in df[isin_col].tolist()]


# -------------------------
# 4) Helpers: nearby bonds by TTM
# -------------------------
def compute_ttm_years(bond_static: pd.DataFrame, asof: pd.Timestamp,
                      maturity_col: str = "maturity_date",
                      ttm_col: str = "ttm_years") -> pd.Series:
    """
    Compute time-to-maturity in years.
    If bond_static already has ttm_years, uses it directly.
    Otherwise computes from maturity_date.
    """
    asof = _as_date(asof)

    if ttm_col in bond_static.columns:
        return pd.to_numeric(bond_static[ttm_col], errors="coerce")

    if maturity_col not in bond_static.columns:
        raise ValueError(f"Need either '{ttm_col}' or '{maturity_col}' in bond_static.")

    mat = pd.to_datetime(bond_static[maturity_col], errors="coerce")
    days = (mat - asof).dt.days.astype("float")
    return days / 365.25


def select_country_curve_bonds(
    bond_static: pd.DataFrame,
    country: str,
    asof: pd.Timestamp,
    min_ttm: float = 1.0,
    max_ttm: float = 32.0,
    isin_col: str = "isin",
    country_col: str = "country",
    maturity_col: str = "maturity_date",
    ttm_col: str = "ttm_years",
) -> pd.DataFrame:
    df = bond_static[bond_static[country_col] == country].copy()
    df["ttm"] = compute_ttm_years(df, asof, maturity_col=maturity_col, ttm_col=ttm_col)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ttm", isin_col])
    df = df[(df["ttm"] >= min_ttm) & (df["ttm"] <= max_ttm)]
    df = df.sort_values("ttm", ascending=True)
    return df


# -------------------------
# 5) Generator protocol
# -------------------------
class UniverseGenerator(Protocol):
    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        ...


# -------------------------
# 6) Concrete generators (one per type family)
# -------------------------
@dataclass
class BenchmarkAcrossTenorsFlyGen:
    countries: Sequence[str]
    tenor_triples: Sequence[Tuple[str, str, str]]  # e.g. [("2Y","5Y","10Y")]
    bond_static: pd.DataFrame
    benchmark_map: pd.DataFrame  # cols: country, tenor, effective_from, effective_to, isin

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            for t1, t2, t3 in self.tenor_triples:
                i1 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c, "tenor": t1})
                i2 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c, "tenor": t2})
                i3 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c, "tenor": t3})
                if not (i1 and i2 and i3):
                    continue

                name = f"{c}_BM_{t1}{t2}{t3}_FLY"
                out.append(
                    Structure(
                        structure_type=StructureType.BM_FLY_ACROSS_TENORS,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=None,
                        tenors=(t1, t2, t3),
                        legs=(Leg(i1, "short"), Leg(i2, "mid"), Leg(i3, "long")),
                        meta={"asof": str(asof.date()), "source": "benchmark_map"},
                    )
                )
        return out


@dataclass
class BenchmarkAcrossTenorsSwitchGen:
    countries: Sequence[str]
    tenor_pairs: Sequence[Tuple[str, str]]  # (short, long) or (left,right) depending your convention
    bond_static: pd.DataFrame
    benchmark_map: pd.DataFrame

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            for t1, t2 in self.tenor_pairs:
                i1 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c, "tenor": t1})
                i2 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c, "tenor": t2})
                if not (i1 and i2):
                    continue

                name = f"{c}_BM_{t1}{t2}_SWITCH"
                out.append(
                    Structure(
                        structure_type=StructureType.BM_SWITCH_ACROSS_TENORS,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=None,
                        tenors=(t1, t2),
                        legs=(Leg(i1, "left"), Leg(i2, "right")),
                        meta={"asof": str(asof.date()), "source": "benchmark_map"},
                    )
                )
        return out


@dataclass
class BenchmarkWithinTenorFlyGen:
    countries: Sequence[str]
    tenors: Sequence[str]  # e.g. ["10Y"]
    bond_static: pd.DataFrame
    benchmark_map: pd.DataFrame  # same mapping table; we use its history by effective_from

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            for tenor in self.tenors:
                hist = resolve_mapping_history(self.benchmark_map, asof, {"country": c, "tenor": tenor})
                # hist[0] is current (OTR), [1] previous, [2] pre-previous
                if len(hist) < 3:
                    continue
                otr, prev, preprev = hist[0], hist[1], hist[2]
                name = f"{c}_BM_{tenor}_ROLL_FLY"
                out.append(
                    Structure(
                        structure_type=StructureType.BM_FLY_WITHIN_TENOR,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=tenor,
                        tenors=None,
                        legs=(Leg(otr, "otr"), Leg(prev, "prev"), Leg(preprev, "preprev")),
                        meta={"asof": str(asof.date()), "source": "benchmark_map_history"},
                    )
                )
        return out


@dataclass
class BenchmarkWithinTenorSwitchGen:
    countries: Sequence[str]
    tenors: Sequence[str]
    bond_static: pd.DataFrame
    benchmark_map: pd.DataFrame

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            for tenor in self.tenors:
                hist = resolve_mapping_history(self.benchmark_map, asof, {"country": c, "tenor": tenor})
                if len(hist) < 2:
                    continue
                otr, prev = hist[0], hist[1]
                name = f"{c}_BM_{tenor}_OTR_PREV_SWITCH"
                out.append(
                    Structure(
                        structure_type=StructureType.BM_SWITCH_WITHIN_TENOR,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=tenor,
                        tenors=None,
                        legs=(Leg(otr, "otr"), Leg(prev, "prev")),
                        meta={"asof": str(asof.date()), "source": "benchmark_map_history"},
                    )
                )
        return out


@dataclass
class NearbyFlyGen:
    countries: Sequence[str]
    bond_static: pd.DataFrame
    min_ttm: float = 1.0
    max_ttm: float = 32.0
    maturity_col: str = "maturity_date"
    ttm_col: str = "ttm_years"
    country_col: str = "country"
    isin_col: str = "isin"
    min_gap_years: float = 0.0  # optional: require ttm[i+2]-ttm[i] >= min_gap_years

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            curve = select_country_curve_bonds(
                self.bond_static, c, asof,
                min_ttm=self.min_ttm, max_ttm=self.max_ttm,
                isin_col=self.isin_col, country_col=self.country_col,
                maturity_col=self.maturity_col, ttm_col=self.ttm_col
            )
            isins = curve[self.isin_col].tolist()
            ttms = curve["ttm"].tolist()

            for i in range(len(isins) - 2):
                if (ttms[i + 2] - ttms[i]) < self.min_gap_years:
                    continue
                a, b, d = isins[i], isins[i + 1], isins[i + 2]
                name = f"{c}_NEARBY_FLY_{i}"
                out.append(
                    Structure(
                        structure_type=StructureType.NEARBY_FLY_ACROSS_CURVE,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=None,
                        tenors=None,
                        legs=(Leg(a, "short"), Leg(b, "mid"), Leg(d, "long")),
                        meta={"asof": str(asof.date()), "method": "sorted_by_ttm"},
                    )
                )
        return out


@dataclass
class NearbySwitchGen:
    countries: Sequence[str]
    bond_static: pd.DataFrame
    min_ttm: float = 1.0
    max_ttm: float = 32.0
    maturity_col: str = "maturity_date"
    ttm_col: str = "ttm_years"
    country_col: str = "country"
    isin_col: str = "isin"
    min_gap_years: float = 0.0  # require ttm[i+1]-ttm[i] >= min_gap_years

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            curve = select_country_curve_bonds(
                self.bond_static, c, asof,
                min_ttm=self.min_ttm, max_ttm=self.max_ttm,
                isin_col=self.isin_col, country_col=self.country_col,
                maturity_col=self.maturity_col, ttm_col=self.ttm_col
            )
            isins = curve[self.isin_col].tolist()
            ttms = curve["ttm"].tolist()

            for i in range(len(isins) - 1):
                if (ttms[i + 1] - ttms[i]) < self.min_gap_years:
                    continue
                a, b = isins[i], isins[i + 1]
                name = f"{c}_NEARBY_SWITCH_{i}"
                out.append(
                    Structure(
                        structure_type=StructureType.NEARBY_SWITCH_ACROSS_CURVE,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=None,
                        tenors=None,
                        legs=(Leg(a, "left"), Leg(b, "right")),
                        meta={"asof": str(asof.date()), "method": "sorted_by_ttm"},
                    )
                )
        return out


@dataclass
class CtdFlyGen:
    countries: Sequence[str]
    future_bucket_triples: Sequence[Tuple[str, str, str]]  # e.g. [("Schatz","Bobl","Bund")]
    bond_static: pd.DataFrame
    ctd_map: pd.DataFrame  # cols: country, future_bucket, effective_from, effective_to, isin

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            for f1, f2, f3 in self.future_bucket_triples:
                i1 = resolve_effective_mapping(self.ctd_map, asof, {"country": c, "future_bucket": f1})
                i2 = resolve_effective_mapping(self.ctd_map, asof, {"country": c, "future_bucket": f2})
                i3 = resolve_effective_mapping(self.ctd_map, asof, {"country": c, "future_bucket": f3})
                if not (i1 and i2 and i3):
                    continue

                name = f"{c}_CTD_{f1}{f2}{f3}_FLY"
                out.append(
                    Structure(
                        structure_type=StructureType.CTD_FLY,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=None,
                        tenors=(f1, f2, f3),
                        legs=(Leg(i1, "short"), Leg(i2, "mid"), Leg(i3, "long")),
                        meta={"asof": str(asof.date()), "source": "ctd_map"},
                    )
                )
        return out


@dataclass
class CtdSwitchGen:
    countries: Sequence[str]
    future_bucket_pairs: Sequence[Tuple[str, str]]
    bond_static: pd.DataFrame
    ctd_map: pd.DataFrame

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c in self.countries:
            for f1, f2 in self.future_bucket_pairs:
                i1 = resolve_effective_mapping(self.ctd_map, asof, {"country": c, "future_bucket": f1})
                i2 = resolve_effective_mapping(self.ctd_map, asof, {"country": c, "future_bucket": f2})
                if not (i1 and i2):
                    continue

                name = f"{c}_CTD_{f1}{f2}_SWITCH"
                out.append(
                    Structure(
                        structure_type=StructureType.CTD_SWITCH,
                        name=name,
                        country=c,
                        countries=None,
                        tenor=None,
                        tenors=(f1, f2),
                        legs=(Leg(i1, "left"), Leg(i2, "right")),
                        meta={"asof": str(asof.date()), "source": "ctd_map"},
                    )
                )
        return out


@dataclass
class CrossCountryBenchmarkSpreadGen:
    country_pairs: Sequence[Tuple[str, str]]  # e.g. [("DE","FR"), ("DE","IT")]
    tenors: Sequence[str]                    # e.g. ["2Y","5Y","10Y","30Y"]
    bond_static: pd.DataFrame
    benchmark_map: pd.DataFrame

    def generate(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        asof = _as_date(asof)

        for c1, c2 in self.country_pairs:
            for tenor in self.tenors:
                i1 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c1, "tenor": tenor})
                i2 = resolve_effective_mapping(self.benchmark_map, asof, {"country": c2, "tenor": tenor})
                if not (i1 and i2):
                    continue

                name = f"{c1}{c2}_BM_{tenor}_SPREAD"
                out.append(
                    Structure(
                        structure_type=StructureType.CROSS_COUNTRY_SPREAD,
                        name=name,
                        country=None,
                        countries=(c1, c2),
                        tenor=tenor,
                        tenors=None,
                        legs=(Leg(i1, f"{c1}"), Leg(i2, f"{c2}")),
                        meta={"asof": str(asof.date()), "source": "benchmark_map"},
                    )
                )
        return out


# -------------------------
# 7) A registry that builds any/all universes
# -------------------------
@dataclass
class UniverseFactory:
    generators: Sequence[UniverseGenerator]

    def generate_all(self, asof: pd.Timestamp) -> List[Structure]:
        out: List[Structure] = []
        for g in self.generators:
            out.extend(g.generate(asof))
        return out

    def generate_by_type(self, asof: pd.Timestamp, types: Sequence[StructureType]) -> List[Structure]:
        all_structs = self.generate_all(asof)
        tset = set(types)
        return [s for s in all_structs if s.structure_type in tset]


# -------------------------
# 8) Convenience: convert to DataFrame for storage/backtesting
# -------------------------
def structures_to_df(structs: Sequence[Structure]) -> pd.DataFrame:
    rows = []
    for s in structs:
        rows.append({
            "type": s.structure_type.name,
            "name": s.name,
            "country": s.country,
            "countries": None if s.countries is None else f"{s.countries[0]}-{s.countries[1]}",
            "tenor": s.tenor,
            "tenors": None if s.tenors is None else ",".join(s.tenors),
            "legs": ";".join([f"{leg.role}:{leg.isin}" for leg in s.legs]),
            "meta": s.meta,
        })
    return pd.DataFrame(rows)

