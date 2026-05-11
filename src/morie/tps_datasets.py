"""morie.tps_datasets — registry + loader for Toronto Police Service
public crime datasets.

13 categories at `data/datasets/TPS/<Category>/CSV/`:
    Assault, AutoTheft, BicycleTheft, BreakandEnter,
    CommunitySafetyIndicators, HateCrimes, Homicides,
    IntimatePartnerAndFamilyViolence, NeighbourhoodCrimeRates, Robbery,
    ShootingAndFirearmDiscarges, TheftFromMovingVehicle, TheftOver

Each category also has Excel/Shapefile/GeoJSON/etc; this module reads
the CSV by default but can be pointed at any sibling format.

Schema (per-incident, varies slightly by category):
    OBJECTID, EVENT_UNIQUE_ID, REPORT_DATE, OCC_DATE,
    REPORT_YEAR, REPORT_MONTH, REPORT_DAY, REPORT_DOY, REPORT_DOW,
    REPORT_HOUR,
    OCC_YEAR, OCC_MONTH, OCC_DAY, OCC_DOY, OCC_DOW, OCC_HOUR,
    DIVISION, LOCATION_TYPE, PREMISES_TYPE,
    UCR_CODE, UCR_EXT, OFFENCE, CSI_CATEGORY,
    HOOD_158, NEIGHBOURHOOD_158, HOOD_140, NEIGHBOURHOOD_140,
    LONG_WGS84, LAT_WGS84, x, y

Public API:
    TPS_REGISTRY        — dict: name → TpsDataset
    load_tps_dataset(name)              -> pd.DataFrame
    list_tps_datasets()                 -> list of (name, n_rows_est, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[5]
TPS_DATA_DIR = PROJECT / "data/datasets/TPS"


@dataclass(frozen=True, slots=True)
class TpsDataset:
    """One TPS open-data category."""
    name: str               # e.g. "Assault"
    description: str
    primary_date: str       # OCC_DATE / REPORT_DATE
    has_geometry: bool      # most do via LAT/LONG_WGS84

    @property
    def csv_dir(self) -> Path:
        return TPS_DATA_DIR / self.name / "CSV"

    def csv_path(self, fname: str | None = None) -> Path:
        d = self.csv_dir
        if fname:
            return d / fname
        # Default: pick the (one) CSV in the dir
        candidates = list(d.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"no CSV in {d}")
        return candidates[0]


TPS_REGISTRY: dict[str, TpsDataset] = {}


def _r(d: TpsDataset) -> None:
    TPS_REGISTRY[d.name] = d


_r(TpsDataset(
    name="Assault",
    description="Reported assault incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="AutoTheft",
    description="Reported auto-theft incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="BicycleTheft",
    description="Reported bicycle thefts in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="BreakandEnter",
    description="Reported break-and-enter incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="CommunitySafetyIndicators",
    description="Toronto community-safety composite indicators",
    primary_date="REPORT_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="HateCrimes",
    description="Reported hate-crime incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="Homicides",
    description="Reported homicide incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="IntimatePartnerAndFamilyViolence",
    description="Reported intimate-partner and family violence in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="NeighbourhoodCrimeRates",
    description="Per-neighbourhood crime rates (annualised, "
                "by HOOD_158)",
    primary_date="REPORT_YEAR", has_geometry=True,
))
_r(TpsDataset(
    name="Robbery",
    description="Reported robbery incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="ShootingAndFirearmDiscarges",
    description="Reported shooting and firearm-discharge incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="TheftFromMovingVehicle",
    description="Reported theft-from-moving-vehicle incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))
_r(TpsDataset(
    name="TheftOver",
    description="Reported theft-over-$5000 incidents in Toronto",
    primary_date="OCC_DATE", has_geometry=True,
))


def load_tps_dataset(name: str,
                     csv_filename: str | None = None,
                     nrows: int | None = None,
                     *, format: str | None = None) -> pd.DataFrame:
    """Load one TPS dataset by category name.

    `name` is case-insensitive. Pass `nrows=N` for a quick sample
    while developing on the largest tables (Assault is 254k rows).

    `format` (optional) — load from a non-CSV format. Valid:
    ``csv``, ``excel``, ``geojson``, ``featurecollection``, ``kml``,
    ``geopackage``, ``sqlitegeodatabase``, ``shapefile``. Default ``csv``.
    Geometric formats attach a ``geometry`` column (Python coord lists).
    See ``morie.tps_io.load_tps`` for details.
    """
    canonical = next((k for k in TPS_REGISTRY if k.lower() == name.lower()),
                     None)
    if canonical is None:
        raise KeyError(
            f"unknown TPS dataset {name!r}. valid: "
            f"{sorted(TPS_REGISTRY.keys())}"
        )
    if format is not None and format.lower() != "csv":
        from .tps_io import load_tps
        df = load_tps(canonical, format=format, nrows=nrows)
    else:
        meta = TPS_REGISTRY[canonical]
        p = meta.csv_path(csv_filename)
        if not p.exists():
            raise FileNotFoundError(
                f"TPS {canonical} CSV not found at {p}. "
                "Verify data/datasets/TPS/<Category>/CSV/ has the export."
            )
        df = pd.read_csv(p, nrows=nrows)
    # Tolerant column-name normalisation for sensitivity-redacted feeds
    # (e.g., HateCrimes uses OCCURRENCE_DATE/YEAR vs OCC_DATE/YEAR and
    # has no LAT_WGS84/LONG_WGS84 because precise geocoding is omitted
    # for privacy):
    rename_map = {}
    for src, dst in [
        ("OCCURRENCE_DATE", "OCC_DATE"),
        ("OCCURRENCE_YEAR", "OCC_YEAR"),
        ("OCCURRENCE_MONTH", "OCC_MONTH"),
        ("OCCURRENCE_DAY", "OCC_DAY"),
        ("OCCURRENCE_HOUR", "OCC_HOUR"),
        ("OCCURRENCE_DOW", "OCC_DOW"),
        ("OCCURRENCE_DOY", "OCC_DOY"),
        ("REPORTED_DATE", "REPORT_DATE"),
        ("REPORTED_YEAR", "REPORT_YEAR"),
    ]:
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def list_tps_datasets() -> list[tuple[str, str, str]]:
    """List all TPS datasets as (name, description, primary_date_col)."""
    return [
        (d.name, d.description, d.primary_date)
        for d in sorted(TPS_REGISTRY.values(), key=lambda x: x.name)
    ]
