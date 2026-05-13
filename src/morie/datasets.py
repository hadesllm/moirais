"""morie.datasets — one-call dataset loaders.

This module wraps the lower-level :mod:`morie.ingest` adapters so that
fresh users can pull common Canadian sociolegal datasets with a single
function call:

  >>> import morie.datasets as md
  >>> df = md.tps_major_crime(year=2024)
  >>> df.shape
  (..., 29)

  >>> df = md.cpads()                   # synthetic fallback if no real PUMF
  >>> df.shape
  (1200, 11)

  >>> idx = md.siu_director_reports()   # may be empty; see siu module docstring
  >>> text = md.siu_report_text("https://www.siu.on.ca/.../22-OFD-001.pdf")

The functions here are **stable user-facing API**.  The underlying
adapters in :mod:`morie.ingest` may change layer URLs / endpoint
shapes as portals reorganise, but the names here will not.

Every function returns a plain :class:`pandas.DataFrame` (or, for
text-mining helpers, a :class:`dict`).  No bespoke result type, no
required configuration, no manual paging.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ----------------------------------------------------------------------
# TPS — Toronto Police Service open data
# ----------------------------------------------------------------------


def tps_major_crime(*, year: int | None = None, max_features: int | None = None,
                    include_geometry: bool = False, offline: bool = False) -> pd.DataFrame:
    """Toronto Police Service "Major Crime Indicators" feed.

    Parameters
    ----------
    year : int | None
        If set, returns only rows with ``OCC_YEAR == year``.  Default
        None returns the entire feed (~700K rows; large).
    max_features : int | None
        Cap on returned rows.  Useful for inspection.
    include_geometry : bool
        If True, includes ``geom_x``/``geom_y`` (web-mercator) columns.
    offline : bool
        If True, returns the bundled 500-row synthetic frame instead
        of hitting the live ArcGIS endpoint.  Use this in CI, on
        air-gapped hosts, or for fast deterministic tests.

    Returns
    -------
    pd.DataFrame with TPS's documented columns (REPORT_DATE, OCC_DATE,
    OFFENCE, MCI_CATEGORY, etc.) — see
    https://data.torontopolice.on.ca/ for the schema.
    """
    if offline:
        import warnings
        from pathlib import Path
        path = Path(__file__).resolve().parent / "data" / "tps_major_crime_synthetic.csv"
        warnings.warn(
            "morie.datasets.tps_major_crime(offline=True): using the bundled "
            "500-row synthetic TPS frame.  This is a toy dataset with the "
            "documented schema but random data; do not interpret outputs "
            "as findings about Toronto crime.",
            UserWarning,
            stacklevel=2,
        )
        df = pd.read_csv(path)
        if year is not None:
            df = df[df["OCC_YEAR"] == year].reset_index(drop=True)
        if max_features is not None:
            df = df.head(max_features)
        return df

    from .ingest.tps import LAYER_REGISTRY, fetch_feature_layer
    where = f"OCC_YEAR = {year}" if year is not None else "1=1"
    return fetch_feature_layer(
        LAYER_REGISTRY["major-crime"],
        where=where,
        max_features=max_features,
        return_geometry=include_geometry,
    )


def otis_a01(*, offline: bool = True) -> pd.DataFrame:
    """Ontario Tracking Information System A01-RCDD (restrictive-confinement)
    correctional-placement records.

    Returns the bundled 800-row synthetic frame by default — real OTIS
    data is FOI-only and cannot be shipped publicly.  Replace with your
    own pull when you have access.

    Schema (canonical morie names):
      person_id, obs_date, fiscal_year, region, alert_complexity,
      volatility_movement, cell_disposition,
      restrictive_confinement_days, age_band.
    """
    import warnings
    from pathlib import Path
    path = Path(__file__).resolve().parent / "data" / "otis_a01_synthetic.csv"
    if offline:
        warnings.warn(
            "morie.datasets.otis_a01(): using the bundled 800-row synthetic "
            "OTIS A01-RCDD frame.  Real OTIS data is FOI-only and cannot be "
            "redistributed; replace with your own load when you have access.",
            UserWarning,
            stacklevel=2,
        )
        return pd.read_csv(path)
    raise NotImplementedError(
        "morie.datasets.otis_a01(offline=False): real OTIS data is FOI-only "
        "and morie cannot fetch it for you.  Pass offline=True for the "
        "synthetic frame, or load your own copy with pandas.read_csv()."
    )


def tps_shootings(*, year: int | None = None, max_features: int | None = None) -> pd.DataFrame:
    """TPS Shooting and Firearm Discharges feed."""
    from .ingest.tps import LAYER_REGISTRY, fetch_feature_layer
    where = f"OCC_YEAR = {year}" if year is not None else "1=1"
    return fetch_feature_layer(LAYER_REGISTRY["shooting-firearms"], where=where,
                               max_features=max_features)


def tps_homicide(*, year: int | None = None, max_features: int | None = None) -> pd.DataFrame:
    """TPS Homicides feed."""
    from .ingest.tps import LAYER_REGISTRY, fetch_feature_layer
    where = f"OCC_YEAR = {year}" if year is not None else "1=1"
    return fetch_feature_layer(LAYER_REGISTRY["homicide"], where=where,
                               max_features=max_features)


def tps_layers() -> pd.DataFrame:
    """List all TPS open-data layers shipped with morie."""
    from .ingest.tps import discover_layers
    return discover_layers()


# ----------------------------------------------------------------------
# CPADS — Canadian Postsecondary Alcohol and Drug Use Survey
# ----------------------------------------------------------------------


def cpads() -> pd.DataFrame:
    """Canadian Postsecondary Alcohol and Drug Use Survey microdata.

    Resolves in this order:
      1. The real Statistics Canada PUMF file if it's available at
         the documented project-root path (data/datasets/oc/CPADS/...).
      2. The 1,200-row shipped-in-wheel synthetic frame with the
         correct schema but random values.

    A warning fires when the synthetic frame is used so toy outputs
    aren't mistaken for findings about the real population.

    Returns
    -------
    pd.DataFrame with morie's canonical CPADS analysis columns
    (weight, alcohol_past12m, heavy_drinking_30d, ebac_tot, ebac_legal,
    cannabis_any_use, age_group, gender, province_region, mental_health,
    physical_health).
    """
    from .modules import load_cpads_analysis_data
    return load_cpads_analysis_data()


# ----------------------------------------------------------------------
# SIU — Special Investigations Unit director's reports
# ----------------------------------------------------------------------


def siu_director_reports() -> pd.DataFrame:
    """SIU director's-reports index.

    NOTE: the SIU re-launched their site in 2025 with a JS-rendered
    case list; this function returns the legacy-pattern PDF anchors
    only, which may be empty.  Use :func:`siu_report_text` with a
    known PDF URL.
    """
    from .ingest.siu import list_reports
    return list_reports()


def siu_report_text(url: str | None = None, *, offline: bool = False) -> str:
    """Download an SIU director's-report PDF and return its plain text.

    Parameters
    ----------
    url : str | None
        Direct PDF URL of a single report.  Required unless ``offline``.
    offline : bool
        If True, returns a bundled synthetic SIU director's-report text
        (case ``24-OFD-001``).  Useful for testing without hitting the
        SIU website.
    """
    if offline:
        from pathlib import Path
        return (Path(__file__).resolve().parent / "data" / "siu_24-OFD-001_synthetic.txt").read_text()
    if url is None:
        raise ValueError("siu_report_text: provide url=... or offline=True")
    from .ingest.siu import fetch_report_text
    return fetch_report_text(url)


def siu_report_fields(text_or_url: str) -> dict[str, Any]:
    """Pull structured fields (sections, conclusion, report_id, date) from
    a report.

    Pass either the text from :func:`siu_report_text` (re-uses it) or a
    PDF URL (fetches first).
    """
    from .ingest.siu import extract_report_fields, fetch_report_text
    if text_or_url.startswith(("http://", "https://")):
        text_or_url = fetch_report_text(text_or_url)
    return extract_report_fields(text_or_url)


# ----------------------------------------------------------------------
# CKAN — generic helper for any open-data CKAN portal
# ----------------------------------------------------------------------


def ckan_search(portal: str, query: str, *, rows: int = 50) -> pd.DataFrame:
    """Search a CKAN open-data portal by free-text query.

    Examples of portals:
      - ``https://open.canada.ca/data``     — Government of Canada
      - ``https://data.gov.uk``             — UK government
      - ``https://data.europa.eu``          — European Union
      - ``https://data.ontario.ca``         — Ontario provincial
    """
    from .ingest.ckan import search_packages
    return search_packages(portal, query, rows=rows)


def ckan_package(portal: str, package_id: str) -> dict[str, pd.DataFrame]:
    """Pull every CSV resource of a CKAN package as a dict of DataFrames.

    Returns a mapping ``{resource_name: DataFrame}``.
    """
    from .ingest.ckan import fetch_package_csvs
    return fetch_package_csvs(portal, package_id)


# ----------------------------------------------------------------------
# What's exported
# ----------------------------------------------------------------------

__all__ = [
    # TPS
    "tps_major_crime",
    "tps_shootings",
    "tps_homicide",
    "tps_layers",
    # CPADS
    "cpads",
    # OTIS
    "otis_a01",
    # SIU
    "siu_director_reports",
    "siu_report_text",
    "siu_report_fields",
    # CKAN
    "ckan_search",
    "ckan_package",
]
