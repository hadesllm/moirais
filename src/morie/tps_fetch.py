# SPDX-License-Identifier: GPL-2.0-only
"""TPS (Toronto Police Service) Open Data ArcGIS REST fetcher.

The Toronto Police Service publishes per-category crime-incident GeoJSON
through ArcGIS Online at services.arcgis.com. Each category has a stable
service layer URL; this module pages through the `/query` endpoint, pulls
all features, and writes a tidy CSV to the morie cache directory.

Public categories (as of 2026-05): Assault, AutoTheft, BicycleTheft,
BreakAndEnter, Homicides, Robbery, ShootingAndFirearmDiscarges,
TheftFromMV, TheftOver, HateCrimes, IntimatePartnerViolence,
CommunitySafetyIndicators, NeighbourhoodCrimeRates.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


__all__ = ["TPS_LAYER_URLS", "fetch_tps_category", "list_tps_categories"]


# Per-category ArcGIS REST FeatureServer layer URLs (layer 0 by convention).
# These are the public layer roots; the /query endpoint is appended per call.
TPS_LAYER_URLS: dict[str, str] = {
    "Assault":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Assault_Open_Data/FeatureServer/0",
    "AutoTheft":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Auto_Theft_Open_Data/FeatureServer/0",
    "BicycleTheft":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Bicycle_Thefts_Open_Data/FeatureServer/0",
    "BreakAndEnter":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Break_and_Enter_Open_Data/FeatureServer/0",
    "Homicides":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Homicides_Open_Data_ASR_RC_TBL_002/FeatureServer/0",
    "Robbery":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Robbery_Open_Data/FeatureServer/0",
    "ShootingAndFirearmDiscarges":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Shooting_and_Firearm_Discharges_Open_Data/FeatureServer/0",
    "TheftFromMV":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Theft_From_Motor_Vehicle_Open_Data/FeatureServer/0",
    "TheftOver":
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/Theft_Over_Open_Data/FeatureServer/0",
}


def list_tps_categories() -> list[str]:
    """Return the list of TPS categories known to this fetcher."""
    return sorted(TPS_LAYER_URLS.keys())


def _arcgis_query(base_url: str, *, where: str, offset: int, max_records: int = 2000) -> dict:
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": max_records,
        "resultOffset": offset,
    }
    url = base_url + "/query?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_tps_category(
    category: str,
    *,
    cache_dir: str | Path = "~/.cache/morie/tps",
    where: str = "1=1",
    overwrite: bool = False,
    max_records_per_page: int = 2000,
) -> Path:
    """Fetch a TPS category as a CSV, paging through ArcGIS until exhausted.

    Args:
        category: One of `list_tps_categories()`.
        cache_dir: Directory to write the CSV into.
        where: SQL `where` clause for the ArcGIS query (default `"1=1"`).
        overwrite: If False and the output file exists, return it without
            re-downloading.
        max_records_per_page: ArcGIS pagination size (server caps at 2000).

    Returns:
        Path to the CSV file in `cache_dir`.

    Raises:
        ValueError: if `category` is unknown.
        urllib.error.URLError: on network failure.
    """
    if category not in TPS_LAYER_URLS:
        raise ValueError(
            f"Unknown TPS category {category!r}. "
            f"Known: {', '.join(list_tps_categories())}"
        )

    out_dir = Path(cache_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tps_{category}.csv"
    if out_path.is_file() and not overwrite:
        return out_path

    base = TPS_LAYER_URLS[category]
    offset = 0
    rows: list[dict] = []
    fieldnames: Optional[list[str]] = None

    while True:
        page = _arcgis_query(base, where=where, offset=offset,
                             max_records=max_records_per_page)
        feats = page.get("features", [])
        if not feats:
            break
        for f in feats:
            props = dict(f.get("properties", {}))
            geom = f.get("geometry") or {}
            if geom.get("type") == "Point" and "coordinates" in geom:
                props.setdefault("LONG_WGS84", geom["coordinates"][0])
                props.setdefault("LAT_WGS84", geom["coordinates"][1])
            if fieldnames is None:
                fieldnames = list(props.keys())
            rows.append(props)
        if len(feats) < max_records_per_page:
            break
        offset += len(feats)

    if not rows:
        raise RuntimeError(f"No features returned for {category!r}")
    fieldnames = fieldnames or list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path
