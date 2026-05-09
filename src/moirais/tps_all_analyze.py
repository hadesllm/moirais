"""moirais.tps_all_analyze — RichResult-emitting analyses for the 13 TPS
crime datasets.

Surfaces:
    analyze_<name>(df=None) -> RichResult        # universal per-dataset
    spatial_summary(df) -> RichResult            # neighbourhood + lat/lon
    temporal_summary(df) -> RichResult           # year/month/dow/hour
    crime_compare(*dfs, names=...) -> RichResult # cross-category
    analyze_all(out_dir=None) -> dict

Output goes to data/manifest/outputs/tps/.

Each TPS CSV has the same standard schema (Toronto Open Data convention),
so a single set of helpers handles the full 13-dataset matrix.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fn._richresult import RichResult
from .tps_datasets import (
    TPS_REGISTRY,
    load_tps_dataset,
)

PROJECT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = PROJECT / "data/manifest/outputs/tps"


# ── Generic helpers ────────────────────────────────────────────────


def _safe_year_col(df: pd.DataFrame) -> str | None:
    for c in ("OCC_YEAR", "REPORT_YEAR", "Year"):
        if c in df.columns:
            return c
    return None


def _vc_rows(s: pd.Series, top: int = 20) -> list[list[Any]]:
    counts = s.value_counts(dropna=False).head(top)
    total = int(counts.sum()) if counts.sum() else 1
    return [
        [str(idx), int(v), f"{100*v/total:.1f}%"]
        for idx, v in counts.items()
    ]


def temporal_summary(df: pd.DataFrame, *, ds_name: str = "?") -> RichResult:
    """Year/month/dow/hour rollups."""
    yc = _safe_year_col(df)
    summary: list[tuple[str, Any]] = [
        ("Dataset", ds_name),
        ("Total incidents", int(df.shape[0])),
    ]
    if yc and df.shape[0]:
        years = pd.to_numeric(df[yc], errors="coerce").dropna()
        if years.size:
            summary.append(("Years covered",
                            f"{int(years.min())}–{int(years.max())}"))
            summary.append(("Mean per year",
                            float(df.shape[0]) / max(1, years.nunique())))

    tables = []
    if yc:
        by_year = (df.groupby(yc).size().sort_index()
                  if df.shape[0] else pd.Series(dtype=int))
        tables.append({
            "title": f"Incidents by {yc}:",
            "headers": [yc, "Count"],
            "rows": [[int(y), int(v)] for y, v in by_year.items()],
        })
    for col, label in [
        ("OCC_MONTH", "Incidents by month of occurrence"),
        ("OCC_DOW", "Incidents by day of week of occurrence"),
        ("OCC_HOUR", "Incidents by hour of occurrence"),
    ]:
        if col in df.columns:
            tables.append({
                "title": f"{label}:",
                "headers": [col, "Count", "Percent"],
                "rows": _vc_rows(df[col], top=24),
            })
    return RichResult(
        title=f"TPS temporal — {ds_name}",
        summary_lines=summary,
        tables=tables,
    )


def spatial_summary(df: pd.DataFrame, *, ds_name: str = "?") -> RichResult:
    """Neighbourhood + division + premises rollups, plus
    lat/long bounding-box summary."""
    summary: list[tuple[str, Any]] = [
        ("Dataset", ds_name),
        ("Incidents", int(df.shape[0])),
    ]
    if "LAT_WGS84" in df.columns and "LONG_WGS84" in df.columns:
        lat = pd.to_numeric(df["LAT_WGS84"], errors="coerce").dropna()
        lon = pd.to_numeric(df["LONG_WGS84"], errors="coerce").dropna()
        if lat.size and lon.size:
            summary.append(("Latitude range",
                            f"{lat.min():.4f}–{lat.max():.4f}"))
            summary.append(("Longitude range",
                            f"{lon.min():.4f}–{lon.max():.4f}"))
            summary.append(("Geocoded incidents", int(min(lat.size, lon.size))))

    tables = []
    if "DIVISION" in df.columns:
        tables.append({
            "title": "Top divisions:",
            "headers": ["Division", "Count", "Percent"],
            "rows": _vc_rows(df["DIVISION"], top=20),
        })
    if "NEIGHBOURHOOD_158" in df.columns:
        tables.append({
            "title": "Top 20 neighbourhoods (158-system):",
            "headers": ["Neighbourhood", "Count", "Percent"],
            "rows": _vc_rows(df["NEIGHBOURHOOD_158"], top=20),
        })
    if "PREMISES_TYPE" in df.columns:
        tables.append({
            "title": "By premises type:",
            "headers": ["Premises", "Count", "Percent"],
            "rows": _vc_rows(df["PREMISES_TYPE"], top=20),
        })
    if "LOCATION_TYPE" in df.columns:
        tables.append({
            "title": "Top 20 location types:",
            "headers": ["Location", "Count", "Percent"],
            "rows": _vc_rows(df["LOCATION_TYPE"], top=20),
        })
    return RichResult(
        title=f"TPS spatial — {ds_name}",
        summary_lines=summary,
        tables=tables,
    )


def offence_summary(df: pd.DataFrame, *, ds_name: str = "?") -> RichResult:
    """OFFENCE / UCR / CSI_CATEGORY rollups."""
    summary = [("Dataset", ds_name), ("Incidents", int(df.shape[0]))]
    tables = []
    for col, label in [
        ("OFFENCE", "Top 20 offences"),
        ("UCR_CODE", "UCR code distribution"),
        ("CSI_CATEGORY", "CSI category distribution"),
    ]:
        if col in df.columns:
            tables.append({
                "title": f"{label}:",
                "headers": [col, "Count", "Percent"],
                "rows": _vc_rows(df[col], top=20),
            })
    return RichResult(
        title=f"TPS offences — {ds_name}",
        summary_lines=summary,
        tables=tables,
    )


# ── Spatial concentration / inequality ─────────────────────────────


def gini_concentration(values: np.ndarray) -> float:
    """Gini coefficient of crime concentration across spatial units.
    G=0 perfectly even, G=1 perfectly concentrated.
    """
    if values.size == 0:
        return float("nan")
    sorted_v = np.sort(values)
    n = sorted_v.size
    cum = np.cumsum(sorted_v, dtype=float)
    if cum[-1] == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


def neighbourhood_concentration(df: pd.DataFrame, *,
                                ds_name: str = "?") -> RichResult:
    """How concentrated is crime across neighbourhoods?
    Uses HOOD_158 (the 158-neighbourhood scheme) by default.
    """
    if "HOOD_158" not in df.columns:
        return RichResult(
            title=f"TPS concentration — {ds_name}",
            warnings=["HOOD_158 column missing — cannot compute "
                      "neighbourhood concentration."],
        )
    hood_counts = df["HOOD_158"].value_counts()
    n_hoods = int(hood_counts.size)
    g = gini_concentration(hood_counts.values.astype(float))
    cum = hood_counts.sort_values(ascending=False).cumsum()
    cum_pct = cum / cum.iloc[-1] if cum.iloc[-1] else cum
    p_top10 = float(cum_pct.iloc[min(9, len(cum_pct) - 1)])
    p_top20 = float(cum_pct.iloc[min(19, len(cum_pct) - 1)])
    return RichResult(
        title=f"TPS neighbourhood concentration — {ds_name}",
        summary_lines=[
            ("Neighbourhoods with ≥1 incident", n_hoods),
            ("Gini coefficient (concentration)", round(g, 4)),
            ("% incidents in top-10 neighbourhoods",
                f"{100*p_top10:.1f}%"),
            ("% incidents in top-20 neighbourhoods",
                f"{100*p_top20:.1f}%"),
        ],
        tables=[{
            "title": "Top 20 neighbourhoods (count, cum %):",
            "headers": ["HOOD_158", "Count", "Cum %"],
            "rows": [[str(h), int(c),
                      f"{100*float(cum_pct.loc[h]):.1f}%"]
                     for h, c in hood_counts.head(20).items()],
        }],
        interpretation=(
            f"Gini = {g:.3f} measures how unequally incidents are "
            "distributed across neighbourhoods. Higher = more "
            f"concentrated. {100*p_top10:.0f}% of incidents fall in "
            "the top-10 neighbourhoods alone."
        ),
        payload={"gini": g, "n_hoods": n_hoods,
                 "p_top10": p_top10, "p_top20": p_top20,
                 "top20": hood_counts.head(20).to_dict()},
    )


# ── Cross-dataset crime compare ────────────────────────────────────


def crime_compare(dfs: dict[str, pd.DataFrame]) -> RichResult:
    """Compare counts and trends across multiple TPS categories.
    Returns one table per axis (year/division/neighbourhood)."""
    rows = []
    for name, df in dfs.items():
        rows.append([name, int(df.shape[0])])
    rows.sort(key=lambda r: -r[1])

    # Year-by-year side-by-side
    year_table = None
    cols = sorted(dfs.keys())
    yc = "OCC_YEAR"
    pivots = []
    for name, df in dfs.items():
        if yc in df.columns:
            s = df.groupby(yc).size()
            s.name = name
            pivots.append(s)
    if pivots:
        m = pd.concat(pivots, axis=1).fillna(0).astype(int).sort_index()
        year_table = {
            "title": "Incidents by OCC_YEAR (side-by-side):",
            "headers": ["OCC_YEAR"] + list(m.columns),
            "rows": [[int(y)] + list(map(int, row))
                     for y, row in m.iterrows()],
        }
    return RichResult(
        title="TPS — cross-category comparison",
        summary_lines=[
            ("Categories compared", len(dfs)),
            ("Total incidents (sum)", sum(df.shape[0] for df in dfs.values())),
        ],
        tables=[
            {"title": "Total counts per category:",
             "headers": ["Category", "Incidents"], "rows": rows},
        ] + ([year_table] if year_table else []),
    )


# ── Per-dataset analysis (uniform across all 13) ───────────────────


def analyze_one(name: str, df: pd.DataFrame | None = None) -> RichResult:
    """Run the standard TPS analysis bundle on one dataset."""
    df = df if df is not None else load_tps_dataset(name)
    return RichResult(
        title=f"TPS {name} — full analysis bundle",
        summary_lines=[
            ("Dataset", name),
            ("Rows", int(df.shape[0])),
            ("Columns", int(df.shape[1])),
        ],
        sections=[
            {"title": "TEMPORAL", "lines": list(temporal_summary(df, ds_name=name).summary_lines)},
            {"title": "SPATIAL", "lines": list(spatial_summary(df, ds_name=name).summary_lines)},
            {"title": "OFFENCES", "lines": list(offence_summary(df, ds_name=name).summary_lines)},
            {"title": "CONCENTRATION", "lines": list(neighbourhood_concentration(df, ds_name=name).summary_lines)},
        ],
    )


# Convenience aliases per category (so users can call analyze_assault())
def _factory(name: str):
    def _fn(df: pd.DataFrame | None = None) -> RichResult:
        return analyze_one(name, df)
    _fn.__name__ = f"analyze_{name.lower()}"
    _fn.__doc__ = f"Analyze TPS {name}."
    return _fn


analyze_assault = _factory("Assault")
analyze_autotheft = _factory("AutoTheft")
analyze_bicycletheft = _factory("BicycleTheft")
analyze_breakandenter = _factory("BreakandEnter")
analyze_communitysafetyindicators = _factory("CommunitySafetyIndicators")
analyze_hatecrimes = _factory("HateCrimes")
analyze_homicides = _factory("Homicides")
analyze_intimatepartnerandfamilyviolence = _factory("IntimatePartnerAndFamilyViolence")
analyze_neighbourhoodcrimerates = _factory("NeighbourhoodCrimeRates")
analyze_robbery = _factory("Robbery")
analyze_shootingandfirearmdiscarges = _factory("ShootingAndFirearmDiscarges")
analyze_theftfrommovingvehicle = _factory("TheftFromMovingVehicle")
analyze_theftover = _factory("TheftOver")


# ── Master driver ──────────────────────────────────────────────────


def analyze_all(out_dir: Path | None = None,
                *, sample_rows: int | None = 50_000) -> dict[str, RichResult]:
    """Run full bundle on each of the 13 TPS datasets + cross-compare.

    `sample_rows` caps each load at N rows for fast development; pass
    None to load everything (Assault alone is 254k rows so the full
    13-way load is several hundred MB).
    """
    out_dir = out_dir or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    loaded: dict[str, pd.DataFrame] = {}
    for name in sorted(TPS_REGISTRY.keys()):
        try:
            df = load_tps_dataset(name, nrows=sample_rows)
            loaded[name] = df
            r = analyze_one(name, df)
            results[name] = r
            (out_dir / f"tps_{name}.txt").write_text(str(r))
            (out_dir / f"tps_{name}.json").write_text(
                json.dumps({"n_rows": int(df.shape[0]),
                            "n_columns": int(df.shape[1])},
                           indent=2)
            )
        except Exception as e:  # noqa: BLE001
            results[name] = RichResult(
                title=f"TPS {name} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )

    # Cross-comparison across all loaded datasets
    if loaded:
        try:
            cross = crime_compare(loaded)
            results["__cross_compare__"] = cross
            (out_dir / "tps_cross_compare.txt").write_text(str(cross))
        except Exception as e:  # noqa: BLE001
            results["__cross_compare__"] = RichResult(
                title="TPS cross-compare (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
