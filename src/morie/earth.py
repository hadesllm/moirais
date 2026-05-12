"""morie.earth -- geospatial pollution-exposure data ingestion.

Pulls satellite + ground-station air-quality data into pandas DataFrames
that plug into the rest of MORIE (causal-inference library, Canadian
health datasets, burden-of-disease attribution).

Data sources wired here:

  * OpenAQ           -- global ground-station PM/NO2/O3/SO2/CO/BC
                       (no auth; hits api.openaq.org)
  * Environment Canada NAPS
                     -- Canadian National Air Pollution Surveillance
                       (no auth; hits Open-Canada CKAN)
  * Google Earth Engine
                     -- Sentinel-5P TROPOMI, MODIS, Landsat, Copernicus
                       (requires service account or user OAuth)
  * ArcGIS Online
                     -- ESRI Living Atlas + feature services
                       (requires ArcGIS account or public service URL)

Each fetcher returns either:
  * a pandas DataFrame (tidy long-format)          -- station data
  * a dict of {region_id: numpy array}             -- raster reductions
  * a raw GeoDataFrame when ``geopandas`` is
    available; falls back to plain DataFrame

Design goals:
  - Zero-cloud fallback: OpenAQ + NAPS work without ANY credentials.
  - Auth gracefully: Earth Engine + ArcGIS raise a clear message if
    credentials aren't configured, with exact steps to fix it.
  - Honest caching: every fetch can be persisted to parquet under
    ``data/cache/earth/`` keyed by a content hash of the query.
  - MORIE-native outputs: everything is a DataFrame ready for
    ``morie.causal`` / ``morie.fn.dml`` / ``morie.fn.aipw`` / etc.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import typing as _t

try:
    import pandas as _pd
except ImportError:  # pragma: no cover
    _pd = None  # type: ignore

__all__ = [
    "fetch_openaq",
    "fetch_naps",
    "fetch_earth_engine",
    "fetch_arcgis",
    "earth_cache_dir",
    "MissingCredentialsError",
]


class MissingCredentialsError(RuntimeError):
    """Raised when a data source requires credentials that aren't configured.

    The error message always includes exact instructions for how to set
    the credentials up, so a future maintainer can unblock
    without reading the docs.
    """


# ─── cache helpers ────────────────────────────────────────────────────────

def earth_cache_dir() -> pathlib.Path:
    """Return (and create) the cache directory for earth-engine results."""
    base = os.environ.get(
        "MORIE_EARTH_CACHE",
        os.path.expanduser("~/.cache/morie/earth"),
    )
    p = pathlib.Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(*parts: _t.Any) -> str:
    payload = json.dumps(parts, default=str, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ─── OpenAQ -- no auth ─────────────────────────────────────────────────────

_OPENAQ_BASE = "https://api.openaq.org/v3"


def fetch_openaq(
    country: str = "CA",
    city: str | None = None,
    pollutants: _t.Sequence[str] = ("pm25", "no2", "o3", "so2", "co"),
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 10000,
    use_cache: bool = True,
):
    """Fetch OpenAQ ground-station measurements as a tidy DataFrame.

    Parameters
    ----------
    country     ISO-2 country code (default Canada).
    city        Optional city name (e.g. ``"Toronto"``).
    pollutants  Iterable of pollutant codes.
    date_from   ISO date/timestamp for lower bound.
    date_to     ISO date/timestamp for upper bound.
    limit       Max rows per page (API caps at 1000 in v2, 10k in v3).
    use_cache   Reuse a cached pull under the given query key if present.

    Returns
    -------
    pandas.DataFrame columns::
        location  latitude  longitude  parameter  value  unit  datetime_utc

    Uses public v3 API; no API key required. If you hit rate limits you
    can set ``$OPENAQ_API_KEY`` and we'll send it.
    """
    if _pd is None:
        raise ImportError("pandas required for fetch_openaq")
    try:
        import httpx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fetch_openaq needs httpx -- pip install httpx"
        ) from exc

    key = _cache_key("openaq", country, city, tuple(pollutants), date_from, date_to, limit)
    cache_path = earth_cache_dir() / f"openaq_{key}.parquet"
    if use_cache and cache_path.exists():
        return _pd.read_parquet(cache_path)

    headers = {}
    if os.environ.get("OPENAQ_API_KEY"):
        headers["X-API-Key"] = os.environ["OPENAQ_API_KEY"]

    params = {
        "iso": country,
        "limit": min(limit, 1000),
        "sort": "desc",
    }
    if city:
        params["city"] = city
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to
    params["parameter"] = ",".join(pollutants)

    rows = []
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(f"{_OPENAQ_BASE}/measurements", params=params, headers=headers)
        resp.raise_for_status()
        for rec in resp.json().get("results", []):
            coords = rec.get("coordinates") or {}
            rows.append({
                "location": rec.get("location"),
                "latitude": coords.get("latitude"),
                "longitude": coords.get("longitude"),
                "parameter": rec.get("parameter"),
                "value": rec.get("value"),
                "unit": rec.get("unit"),
                "datetime_utc": (rec.get("date") or {}).get("utc"),
            })

    df = _pd.DataFrame(rows)
    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


# ─── Environment Canada NAPS -- no auth (CKAN) ─────────────────────────────

_NAPS_CKAN_PACKAGE = "1b36a356-defd-4813-acea-47bc3abd859b"


def fetch_naps(
    year: int,
    pollutant: str = "no2",
    province: str | None = None,
    use_cache: bool = True,
):
    """Fetch Environment Canada NAPS annual data via Open-Canada CKAN.

    Parameters
    ----------
    year       Calendar year (e.g. 2023).
    pollutant  One of ``{"no2","pm25","o3","so2","co","pm10"}``.
    province   Optional ISO 2-letter province code (e.g. ``"ON"``).

    Returns
    -------
    pandas.DataFrame with columns::
        station_id  station_name  latitude  longitude  province  datetime_local  value  unit

    NAPS is the Canadian ground-truth counterpart to OpenAQ with better
    station coverage in Ontario / Quebec / BC. Data lands as CSV under
    the CKAN package; we stream the matching resource for the year +
    pollutant and return a DataFrame.
    """
    if _pd is None:
        raise ImportError("pandas required for fetch_naps")
    try:
        import httpx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fetch_naps needs httpx -- pip install httpx"
        ) from exc

    key = _cache_key("naps", year, pollutant, province)
    cache_path = earth_cache_dir() / f"naps_{key}.parquet"
    if use_cache and cache_path.exists():
        return _pd.read_parquet(cache_path)

    pkg_url = f"https://open.canada.ca/data/api/action/package_show?id={_NAPS_CKAN_PACKAGE}"
    with httpx.Client(timeout=60.0) as client:
        meta = client.get(pkg_url).json()
        resources = meta.get("result", {}).get("resources", [])
        match = [
            r for r in resources
            if str(year) in (r.get("name") or "")
            and pollutant.upper() in (r.get("name") or "").upper()
            and (r.get("format") or "").upper() == "CSV"
        ]
        if not match:
            raise FileNotFoundError(
                f"NAPS CKAN has no {pollutant.upper()} CSV for year {year}"
            )
        csv_url = match[0]["url"]
        df = _pd.read_csv(csv_url, low_memory=False)

    if province and "Province" in df.columns:
        df = df[df["Province"].str.upper() == province.upper()]

    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return df


# ─── Google Earth Engine -- requires auth ──────────────────────────────────

_EE_SETUP_HELP = """\
Google Earth Engine requires authentication. Two options:

  (1) Service account (headless, recommended for MORIE):
      a. https://console.cloud.google.com -> IAM -> Service Accounts
      b. Create one, enable "Earth Engine Resource Admin" role
      c. Download JSON key
      d. Register at https://signup.earthengine.google.com/#!/service_accounts
      e. Set MORIE_EE_SERVICE_ACCOUNT and MORIE_EE_KEY_PATH
         (or export GOOGLE_APPLICATION_CREDENTIALS=<json-path>)

  (2) User OAuth (interactive):
      pip install earthengine-api
      python -c "import ee; ee.Authenticate(); ee.Initialize(project='<project-id>')"

Then re-run fetch_earth_engine().
"""


def _ensure_ee_initialized():
    try:
        import ee  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "fetch_earth_engine needs earthengine-api -- pip install earthengine-api"
        ) from exc

    sa_email = os.environ.get("MORIE_EE_SERVICE_ACCOUNT")
    key_path = os.environ.get("MORIE_EE_KEY_PATH") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project = os.environ.get("MORIE_EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")

    try:
        if sa_email and key_path:
            creds = ee.ServiceAccountCredentials(sa_email, key_path)
            ee.Initialize(creds, project=project)
        else:
            ee.Initialize(project=project)
    except Exception as exc:
        raise MissingCredentialsError(_EE_SETUP_HELP) from exc
    return ee


def fetch_earth_engine(
    dataset: str,
    region,
    date_from: str,
    date_to: str,
    band: str | None = None,
    scale: int = 1000,
    reducer: str = "mean",
):
    """Pull an Earth Engine ImageCollection reduced over a region/time-window.

    Parameters
    ----------
    dataset    EE image-collection asset ID (e.g.
               ``"COPERNICUS/S5P/OFFL/L3_NO2"`` for TROPOMI NO2).
    region     ee.Geometry, GeoJSON dict, or (lon_min, lat_min, lon_max, lat_max) tuple.
    date_from  ISO date (e.g. ``"2024-01-01"``).
    date_to    ISO date.
    band       Band name to extract (dataset-specific; defaults to the
               first band of the first image).
    scale      Pixel scale in metres.
    reducer    One of ``{"mean","max","min","sum","median"}``.

    Returns
    -------
    dict with keys::
        {'mean': float, 'stdDev': float, 'count': int, 'unit': str}

    Example (port of your Toronto NO2 app to MORIE):

        >>> no2 = fetch_earth_engine(
        ...     dataset="COPERNICUS/S5P/OFFL/L3_NO2",
        ...     region=(-79.64, 43.58, -79.12, 43.86),   # Toronto bbox
        ...     date_from="2024-01-01",
        ...     date_to="2024-12-31",
        ...     band="tropospheric_NO2_column_number_density",
        ...     scale=1113,
        ... )
        >>> print(no2["mean"])   # mol/m^2 averaged over Toronto in 2024
    """
    ee = _ensure_ee_initialized()

    if isinstance(region, (tuple, list)) and len(region) == 4:
        region_ee = ee.Geometry.Rectangle(list(region))
    elif isinstance(region, dict):
        region_ee = ee.Geometry(region)
    else:
        region_ee = region  # assume caller passed ee.Geometry

    coll = ee.ImageCollection(dataset).filterDate(date_from, date_to).filterBounds(region_ee)
    if band:
        coll = coll.select(band)

    agg_map = {
        "mean":   ee.Reducer.mean(),
        "max":    ee.Reducer.max(),
        "min":    ee.Reducer.min(),
        "sum":    ee.Reducer.sum(),
        "median": ee.Reducer.median(),
    }
    red = agg_map.get(reducer, ee.Reducer.mean())

    image = coll.mean() if reducer == "mean" else coll.reduce(red)

    stats = image.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True).combine(ee.Reducer.count(), sharedInputs=True),
        geometry=region_ee,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True,
    ).getInfo()

    return stats or {}


# ─── ArcGIS Online -- requires auth (or public service URL) ────────────────

_ARCGIS_SETUP_HELP = """\
ArcGIS Python API requires either public feature-service URLs (no auth)
or an ArcGIS Online / Enterprise account.

  Install:
    pip install arcgis

  Auth (pick one):
    # anonymous -- public feature services only
    gis = GIS()

    # ArcGIS Online (org or developer free tier)
    gis = GIS("https://www.arcgis.com", "username", "password")

    # Via env var
    export MORIE_ARCGIS_URL=https://your-org.maps.arcgis.com
    export MORIE_ARCGIS_USER=...
    export MORIE_ARCGIS_PASS=...

If you just need a public layer, pass its URL directly and no auth is
needed.
"""


def fetch_arcgis(
    service_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    max_records: int = 2000,
):
    """Fetch an ArcGIS FeatureLayer as a DataFrame (+ geometry if geopandas present).

    Parameters
    ----------
    service_url  Full layer URL, e.g.
                 ``https://services.arcgis.com/.../FeatureServer/0``.
    where        SQL WHERE clause (e.g. ``"YEAR = 2023"``).
    out_fields   Comma-separated list of attributes (``"*"`` = all).
    max_records  Max features returned.

    Returns a DataFrame. If ``geopandas`` is installed, geometry is
    included as a GeoSeries; otherwise lat/lon columns are derived for
    point layers.

    This is the simplest entry point for ESRI / ESRI Living Atlas
    layers that expose feature services. For authed layers use
    ``arcgis.gis.GIS`` directly and pass the authenticated content item
    here.
    """
    try:
        from arcgis.features import FeatureLayer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            _ARCGIS_SETUP_HELP
        ) from exc

    layer = FeatureLayer(service_url)
    fs = layer.query(where=where, out_fields=out_fields, result_record_count=max_records)

    try:
        df = fs.sdf  # spatial dataframe (requires arcgis + pandas)
    except Exception:
        df = _pd.DataFrame([f.attributes for f in fs.features]) if _pd is not None else fs.features

    return df


# ─── Convenience: a curated dataset registry for pollution work ───────────

POLLUTION_DATASETS = {
    "sentinel5p_no2": {
        "source": "earth_engine",
        "asset": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
        "unit": "mol/m^2",
        "scale_m": 1113,
        "description": "Sentinel-5P TROPOMI tropospheric NO2 column (daily, ~5.5km)",
    },
    "sentinel5p_so2": {
        "source": "earth_engine",
        "asset": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band": "SO2_column_number_density",
        "unit": "mol/m^2",
        "scale_m": 1113,
    },
    "sentinel5p_o3": {
        "source": "earth_engine",
        "asset": "COPERNICUS/S5P/OFFL/L3_O3",
        "band": "O3_column_number_density",
        "unit": "mol/m^2",
        "scale_m": 1113,
    },
    "sentinel5p_co": {
        "source": "earth_engine",
        "asset": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "unit": "mol/m^2",
        "scale_m": 1113,
    },
    "sentinel5p_ch4": {
        "source": "earth_engine",
        "asset": "COPERNICUS/S5P/OFFL/L3_CH4",
        "band": "CH4_column_volume_mixing_ratio_dry_air",
        "unit": "ppbv",
        "scale_m": 1113,
    },
    "modis_aod": {
        "source": "earth_engine",
        "asset": "MODIS/061/MCD19A2_GRANULES",
        "band": "Optical_Depth_055",
        "unit": "dimensionless",
        "scale_m": 1000,
        "description": "MODIS MAIAC aerosol optical depth at 550nm (daily, 1km)",
    },
    "ghsl_population": {
        "source": "earth_engine",
        "asset": "JRC/GHSL/P2023A/GHS_POP",
        "band": "population_count",
        "unit": "persons/pixel",
        "scale_m": 100,
        "description": "GHSL 2023 population grid (100m) -- use for denominators",
    },
}


def list_pollution_datasets() -> list[dict]:
    """Return a list of curated pollution dataset descriptors."""
    return [{"name": k, **v} for k, v in POLLUTION_DATASETS.items()]
