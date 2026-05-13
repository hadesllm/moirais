"""Toronto Police Service open-data client.

The Toronto Police Service publishes its open-data feeds through an
ArcGIS Hub at https://data.torontopolice.on.ca/ — every dataset is
backed by an ArcGIS FeatureServer with a stable REST API that
returns GeoJSON or JSON depending on the request.

This module is a thin convenience layer over those endpoints that
returns plain :class:`pandas.DataFrame` objects (geometry dropped or
preserved at the caller's choice) so morie's MRM modules can ingest
TPS feeds with one call.

Quick usage
-----------

  >>> from morie.ingest.tps import fetch_feature_layer
  >>> df = fetch_feature_layer(MAJOR_CRIME_LAYER, where="OCC_YEAR >= 2023")
  >>> df.shape
  (N, K)

  >>> from morie.ingest.tps import discover_layers
  >>> discover_layers()
  pd.DataFrame(...)  # all published TPS layers

CLI
---

::

    morie ingest tps --layer major-crime --year 2024 --out tps-major-2024.csv

"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

DEFAULT_USER_AGENT = "morie/0.5.0 (+https://github.com/hadesllm/morie)"
DEFAULT_TIMEOUT_SECONDS = 60.0

# Canonical TPS open-data layer endpoints.  These IDs are stable
# (verified 2026-05-13) but the underlying service URLs can rotate
# under ArcGIS Hub re-organisation; fall back to discover_layers()
# if an entry below 404s.
LAYER_REGISTRY: dict[str, str] = {
    "major-crime": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Major_Crime_Indicators_Open_Data/FeatureServer/0"
    ),
    "shooting-firearms": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Shooting_and_Firearm_Discharges_Open_Data/FeatureServer/0"
    ),
    "homicide": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Homicides_Open_Data/FeatureServer/0"
    ),
    "robbery": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Robbery_Open_Data/FeatureServer/0"
    ),
    "assault": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Assault_Open_Data/FeatureServer/0"
    ),
    "auto-theft": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Auto_Theft_Open_Data/FeatureServer/0"
    ),
    "break-and-enter": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Break_and_Enter_Open_Data/FeatureServer/0"
    ),
    "theft-over": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Theft_Over_Open_Data/FeatureServer/0"
    ),
    "bicycle-thefts": (
        "https://services.arcgis.com/S9th0jAJ7bqgIRjw/arcgis/rest/services/"
        "Bicycle_Thefts_Open_Data/FeatureServer/0"
    ),
}

MAJOR_CRIME_LAYER = LAYER_REGISTRY["major-crime"]


class TPSError(RuntimeError):
    """A TPS ArcGIS endpoint returned an HTTP error or no features."""


def _arcgis_query(
    layer_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = False,
    result_offset: int = 0,
    result_record_count: int = 2000,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Single ArcGIS FeatureServer /query call."""
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": str(return_geometry).lower(),
        "resultOffset": result_offset,
        "resultRecordCount": result_record_count,
        "f": "json",
    }
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        r = c.get(f"{layer_url}/query", params=params)
        if r.status_code >= 400:
            raise TPSError(f"layer query -> HTTP {r.status_code}: {r.text[:200]}")
        payload = r.json()
        if "error" in payload:
            raise TPSError(f"layer query error: {payload['error']}")
        return payload


def fetch_feature_layer(
    layer_url: str,
    *,
    where: str = "1=1",
    out_fields: str = "*",
    return_geometry: bool = False,
    page_size: int = 2000,
    max_features: int | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch every feature from a TPS open-data layer.

    ArcGIS FeatureServer queries cap at 2,000 records per call (the
    server-side ``maxRecordCount``).  This function pages through the
    layer transparently, emitting one DataFrame.

    Parameters
    ----------
    layer_url : str
        Full URL to the FeatureServer layer, e.g. one of the entries
        in :data:`LAYER_REGISTRY`.
    where : str
        ArcGIS WHERE clause.  Default "1=1" fetches everything.
        Examples: ``"OCC_YEAR=2024"``, ``"OCC_YEAR BETWEEN 2020 AND 2025"``.
    out_fields : str
        Comma-separated attribute list, or "*" for all.
    return_geometry : bool
        If True, includes ``geometry.x`` / ``geometry.y`` columns.
    page_size : int
        Records per request; clamped server-side to 2,000.
    max_features : int | None
        Optional hard cap on total returned rows.
    timeout : float
        Per-request timeout in seconds.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = _arcgis_query(
            layer_url,
            where=where,
            out_fields=out_fields,
            return_geometry=return_geometry,
            result_offset=offset,
            result_record_count=page_size,
            timeout=timeout,
        )
        features = payload.get("features", [])
        if not features:
            break
        for f in features:
            row = dict(f.get("attributes") or {})
            if return_geometry and f.get("geometry"):
                row["geom_x"] = f["geometry"].get("x")
                row["geom_y"] = f["geometry"].get("y")
            rows.append(row)
        if max_features is not None and len(rows) >= max_features:
            rows = rows[:max_features]
            break
        if not payload.get("exceededTransferLimit", False):
            break
        offset += len(features)

    if not rows:
        raise TPSError(f"layer returned zero features: where={where!r}")
    return pd.DataFrame(rows)


def discover_layers() -> pd.DataFrame:
    """Return the built-in TPS layer registry as a DataFrame.

    Use this to discover layer names without leaving Python; pass an
    entry's ``url`` field to :func:`fetch_feature_layer`.
    """
    return pd.DataFrame(
        [{"name": k, "url": v} for k, v in LAYER_REGISTRY.items()]
    )


# ----------------------------------------------------------------------
# CLI handler — wired into morie/runner.py


def cli(args: list[str]) -> int:
    """Handle ``morie ingest tps ...``.  Returns exit code."""
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="morie ingest tps",
                                description="Pull a Toronto Police Service open-data layer.")
    p.add_argument("--layer", help=f"Built-in layer name: {sorted(LAYER_REGISTRY)}")
    p.add_argument("--url", help="Direct FeatureServer layer URL (overrides --layer)")
    p.add_argument("--year", type=int, help="Filter to a single OCC_YEAR")
    p.add_argument("--where", help="Raw ArcGIS WHERE clause (overrides --year)")
    p.add_argument("--max", type=int, dest="max_features", help="Cap returned rows")
    p.add_argument("--geometry", action="store_true", help="Include geom_x / geom_y columns")
    p.add_argument("--out", type=Path, help="CSV output path (stdout if omitted)")
    p.add_argument("--list", action="store_true", help="List the built-in layer names and exit")
    ns = p.parse_args(args)

    if ns.list:
        sys.stdout.write(discover_layers().to_csv(index=False))
        return 0

    url = ns.url
    if url is None and ns.layer:
        if ns.layer not in LAYER_REGISTRY:
            sys.stderr.write(f"unknown layer {ns.layer!r}; try --list\n")
            return 2
        url = LAYER_REGISTRY[ns.layer]
    if url is None:
        p.error("provide --layer or --url")
        return 2

    where = ns.where or (f"OCC_YEAR = {ns.year}" if ns.year else "1=1")
    df = fetch_feature_layer(url, where=where, return_geometry=ns.geometry,
                             max_features=ns.max_features)
    if ns.out:
        df.to_csv(ns.out, index=False)
        sys.stderr.write(f"wrote {ns.out}  ({len(df):,} rows, {len(df.columns)} cols)\n")
    else:
        sys.stdout.write(df.to_csv(index=False))
    return 0
