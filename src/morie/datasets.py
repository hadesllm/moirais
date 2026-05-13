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
# Chicago — City of Chicago open data (Socrata)
# ----------------------------------------------------------------------


# Documented Socrata column schema for the Chicago "Crimes — 2001 to
# Present" feed (resource ijzp-q8t2).  Kept here verbatim so the
# offline fallback can synthesise a frame with the same columns even
# when the bundled CSV doesn't ship every field.
_CHICAGO_CRIME_COLUMNS: tuple[str, ...] = (
    "id", "case_number", "date", "block", "iucr", "primary_type",
    "description", "location_description", "arrest", "domestic",
    "beat", "district", "ward", "community_area", "fbi_code",
    "x_coordinate", "y_coordinate", "year", "updated_on",
    "latitude", "longitude",
)


def chicago_crime(*, year: int | None = None, max_features: int | None = None,
                  offline: bool = False) -> pd.DataFrame:
    """City of Chicago "Crimes — 2001 to Present" feed.

    Schema (Socrata column names, preserved verbatim):
      id, case_number, date, block, iucr, primary_type, description,
      location_description, arrest, domestic, beat, district, ward,
      community_area, fbi_code, x_coordinate, y_coordinate, year,
      updated_on, latitude, longitude.

    Parameters
    ----------
    year : int | None
        Server-side filter to a single ``year`` (e.g. 2024).  ``None``
        returns the entire feed (millions of rows; large).
    max_features : int | None
        Cap on returned rows.  Useful for inspection.
    offline : bool
        If True, returns the bundled tiny synthetic Chicago frame
        instead of hitting the live Socrata endpoint.  Use this in CI,
        on air-gapped hosts, or for fast deterministic tests.

    Notes
    -----
    The live endpoint is `https://data.cityofchicago.org/resource/ijzp-q8t2.json`
    (resource id ``ijzp-q8t2``).  Anonymous calls share a throttled
    rate-limit pool; for production pulls register a Socrata
    application token and pass it via the lower-level
    :func:`morie.ingest.chicago.fetch_crime` helper.
    """
    if offline:
        import warnings
        from pathlib import Path
        path = Path(__file__).resolve().parent / "data" / "chicago_crime_synthetic.csv"
        warnings.warn(
            "morie.datasets.chicago_crime(offline=True): using the bundled "
            "synthetic Chicago crime frame.  This is a toy dataset with the "
            "documented Socrata schema but random data; do not interpret "
            "outputs as findings about Chicago crime.",
            UserWarning,
            stacklevel=2,
        )
        if path.exists():
            df = pd.read_csv(path)
        else:
            # Empty frame with the right shape so downstream code that
            # only inspects ``df.columns`` keeps working even before
            # the synthetic CSV lands in data/.
            df = pd.DataFrame({c: [] for c in _CHICAGO_CRIME_COLUMNS})
        if year is not None and "year" in df.columns and len(df) > 0:
            df = df[df["year"] == year].reset_index(drop=True)
        if max_features is not None:
            df = df.head(max_features)
        return df

    from .ingest.chicago import fetch_crime
    return fetch_crime(year=year, max_features=max_features)


# ----------------------------------------------------------------------
# NYC — NYC OpenData (Socrata)
# ----------------------------------------------------------------------


# NYC Stop, Question and Frisk (SQF) is published as one Socrata
# resource per release year.  The mapping below is the current
# (verified 2026-05-13) NYC OpenData id for each year; we expose the
# most-recent entry as the default and raise a clear ValueError if
# the caller asks for a year we don't have an id for yet.
#
# Schema across years is broadly consistent (stop_id, stop_date,
# precinct, suspect_race, suspect_sex, suspect_age, frisked, searched,
# summons_issued, arrest_made, ...) but NYPD has rotated field names
# between releases; the loader does NOT rename columns, so downstream
# code should treat the column list as year-specific.
_NYC_SQF_RESOURCES: dict[int, str] = {
    # 2022 onwards live under the modern release; older years used
    # different ids and are intentionally not pre-registered here —
    # callers needing 2003–2017 can pass an explicit resource URL to
    # :func:`morie.ingest.chicago.fetch_socrata` directly.
    2024: "https://data.cityofnewyork.us/resource/7v9w-k82r.json",
    2023: "https://data.cityofnewyork.us/resource/rbed-zzin.json",
    2022: "https://data.cityofnewyork.us/resource/e4yi-bvqr.json",
}

_NYC_SQF_DEFAULT_YEAR = max(_NYC_SQF_RESOURCES)


def nyc_stop_and_frisk(*, year: int | None = None, max_features: int | None = None,
                       offline: bool = False) -> pd.DataFrame:
    """NYPD Stop, Question and Frisk (SQF) microdata via NYC OpenData.

    NYC publishes SQF as a separate Socrata resource per release year.
    Pass ``year`` to pick which release to pull; ``None`` defaults to
    the most-recent registered year (currently 2024).

    Parameters
    ----------
    year : int | None
        Release year.  Must be a key of the internal resource map
        (currently 2022–2024 inclusive).  ``None`` → most recent.
    max_features : int | None
        Cap on returned rows.
    offline : bool
        If True, returns the bundled synthetic NYC SQF frame.

    Notes
    -----
    Schema is NOT normalised across years — NYPD has rotated field
    names between releases.  Inspect ``df.columns`` to see the exact
    schema of the requested release.
    """
    if offline:
        import warnings
        from pathlib import Path
        path = Path(__file__).resolve().parent / "data" / "nyc_sqf_synthetic.csv"
        warnings.warn(
            "morie.datasets.nyc_stop_and_frisk(offline=True): using the "
            "bundled synthetic NYC SQF frame.  This is a toy dataset with a "
            "schematic SQF column layout but random data; do not interpret "
            "outputs as findings about NYC stop-and-frisk.",
            UserWarning,
            stacklevel=2,
        )
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()
        if max_features is not None and len(df) > 0:
            df = df.head(max_features)
        return df

    chosen_year = _NYC_SQF_DEFAULT_YEAR if year is None else int(year)
    if chosen_year not in _NYC_SQF_RESOURCES:
        raise ValueError(
            f"morie.datasets.nyc_stop_and_frisk: no built-in NYC OpenData "
            f"resource for year={chosen_year}.  Known years: "
            f"{sorted(_NYC_SQF_RESOURCES)}.  Pass a custom URL to "
            f"morie.ingest.chicago.fetch_socrata() for older releases."
        )
    from .ingest.chicago import fetch_socrata
    return fetch_socrata(
        _NYC_SQF_RESOURCES[chosen_year],
        max_features=max_features,
    )


# ----------------------------------------------------------------------
# BigQuery — Google BigQuery public-data adapter (optional dep)
# ----------------------------------------------------------------------


def bigquery(project: str, dataset: str, table: str, *,
             where: str | None = None, limit: int | None = None,
             select: str = "*",
             billing_project: str | None = None) -> pd.DataFrame:
    """Pull a BigQuery table (or filtered slice) as a DataFrame.

    Lightweight wrapper around :mod:`morie.ingest.bigquery` for the
    common "give me this public table, optionally filtered" case.

    Parameters
    ----------
    project, dataset, table : str
        Fully-qualified table, e.g. ``project="bigquery-public-data"``,
        ``dataset="chicago_crime"``, ``table="crime"``.
    where : str | None
        Raw SQL ``WHERE`` clause (without the leading ``WHERE``).
    limit : int | None
        Optional ``LIMIT``.
    select : str
        Projection list; defaults to ``*``.
    billing_project : str | None
        GCP project to bill the query to.  Public datasets cost the
        *caller's* project, not the dataset owner.  ``None`` uses the
        ADC-discovered project.

    Notes
    -----
    Requires the ``bigquery`` extra:

    .. code-block:: shell

        pip install 'morie[bigquery]'

    Authenticates via Application Default Credentials (ADC), matching
    the HADES-LLM Pi-rendered architecture.
    """
    from .ingest.bigquery import fetch_table
    return fetch_table(
        project=project, dataset=dataset, table=table,
        where=where, limit=limit, select=select,
        billing_project=billing_project,
    )


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
# Forensics — US federal forensic open-data
# ----------------------------------------------------------------------


def _forensics_synthetic(name: str, *, kind: str) -> pd.DataFrame:
    """Load the bundled synthetic frame for a forensics endpoint.

    Raises a clean :class:`FileNotFoundError` (rather than a cryptic
    pandas error) if the toy CSV hasn't been generated yet, so callers
    in dev / CI can tell the difference between "missing fallback" and
    "live API down".
    """
    import warnings
    from pathlib import Path

    path = Path(__file__).resolve().parent / "data" / f"{name}_synthetic.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"morie.datasets.{kind}(offline=True): no bundled synthetic "
            f"frame at {path}.  Run the smoke-test generator to create "
            f"one, or call with offline=False."
        )
    warnings.warn(
        f"morie.datasets.{kind}(offline=True): using the bundled "
        f"synthetic {kind} frame.  This is a toy dataset with the "
        f"documented schema but random data; do not interpret outputs "
        f"as findings about real forensic populations.",
        UserWarning,
        stacklevel=3,
    )
    return pd.read_csv(path)


def nibrs(
    *,
    year: int | None = None,
    max_features: int | None = None,
    state: str | None = None,
    offense: str | None = None,
    api_key: str | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """FBI National Incident-Based Reporting System (NIBRS) offence events.

    Pulls offence-event records from the FBI Crime Data Explorer
    (``https://api.usa.gov/crime/fbi/cde/``).  An API key is required;
    sign up free at ``https://api.data.gov/signup/`` and pass via
    ``api_key=`` or export ``FBI_CDE_API_KEY``.

    Parameters
    ----------
    year : int | None
        Reporting year (required unless ``offline=True``).
    max_features : int | None
        Cap on returned rows.  Recommended for the national feed.
    state : str | None
        Two-letter US state code (e.g. ``"GA"``).  None = national.
    offense : str | None
        NIBRS offence slug (e.g. ``"aggravated-assault"``).  None = all.
    api_key : str | None
        FBI CDE API key.  Falls back to ``$FBI_CDE_API_KEY``.
    offline : bool
        If True, returns a bundled synthetic NIBRS frame for CI/airgap.

    Returns
    -------
    pd.DataFrame with one row per offence-event; nested NIBRS sub-objects
    flattened to dotted-key columns (``offense.code``, ``victim.age``).
    """
    if offline:
        df = _forensics_synthetic("nibrs", kind="nibrs")
        if max_features is not None:
            df = df.head(max_features)
        return df

    if year is None:
        raise ValueError(
            "morie.datasets.nibrs: year=... is required unless offline=True"
        )

    from .ingest.forensics import fetch_nibrs
    return fetch_nibrs(
        year=year, offense=offense, state=state, api_key=api_key,
        max_features=max_features,
    )


def namus_missing_persons(
    *,
    state: str | None = None,
    max_features: int | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """NamUs missing-persons case metadata.

    Pulls case metadata from the National Missing and Unidentified
    Persons System (``https://www.namus.gov/``).  No API key required.

    Parameters
    ----------
    state : str | None
        Two-letter US state code (e.g. ``"CA"``).  None = national.
    max_features : int | None
        Cap on returned rows.
    offline : bool
        If True, returns a bundled synthetic NamUs frame.

    Returns
    -------
    pd.DataFrame with columns: case_number, state, county, dlc_date, sex,
    race, age_min, age_max, height_cm_min, height_cm_max, weight_kg_min,
    weight_kg_max, first_name, last_name, city, circumstances.
    """
    if offline:
        df = _forensics_synthetic(
            "namus_missing_persons", kind="namus_missing_persons"
        )
        if state is not None and "state" in df.columns:
            df = df[df["state"].astype(str).str.upper() == state.upper()].reset_index(drop=True)
        if max_features is not None:
            df = df.head(max_features)
        return df

    from .ingest.forensics import fetch_namus_missing_persons
    return fetch_namus_missing_persons(state=state, max_features=max_features)


def nist_rds(
    *,
    dataset_id: str | None = None,
    query: str | None = None,
    max_features: int | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """NIST Reference Datasets catalog metadata.

    Returns the *catalog* records for NIST RDS holdings (CSAFE, NSRL,
    ...).  The reference datasets themselves are multi-gigabyte and
    served on dedicated download servers — pull them out-of-band using
    the ``landing_page`` column.

    Parameters
    ----------
    dataset_id : str | None
        Specific NIST RDS / EDI id; returns a single-row frame.
    query : str | None
        Free-text search over title/description/keyword.
    max_features : int | None
        Cap on returned rows.
    offline : bool
        If True, returns a bundled synthetic NIST RDS catalog frame.

    Returns
    -------
    pd.DataFrame with columns: dataset_id, title, description, publisher,
    issued, modified, keyword, landing_page, size_bytes, license.
    """
    if offline:
        df = _forensics_synthetic("nist_rds", kind="nist_rds")
        if dataset_id is not None and "dataset_id" in df.columns:
            df = df[df["dataset_id"].astype(str) == dataset_id].reset_index(drop=True)
        if max_features is not None:
            df = df.head(max_features)
        return df

    from .ingest.forensics import fetch_nist_rds
    return fetch_nist_rds(
        dataset_id=dataset_id, query=query, max_features=max_features,
    )


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
    # Chicago / NYC (Socrata)
    "chicago_crime",
    "nyc_stop_and_frisk",
    # BigQuery (optional dep)
    "bigquery",
    # CKAN
    "ckan_search",
    "ckan_package",
    # Forensics (US)
    "nibrs",
    "namus_missing_persons",
    "nist_rds",
]
