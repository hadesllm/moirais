"""City of Chicago open-data client (Socrata-style).

The City of Chicago publishes its open data through a Socrata-powered
portal at https://data.cityofchicago.org/.  Every dataset is backed by
a stable Socrata Open Data API (SODA) endpoint that returns JSON, with
SoQL ($where / $select / $limit / $offset) for server-side filtering
and paging.

This module is a thin convenience layer over those endpoints — and
over Socrata portals generally, since the API shape is shared with
NYC OpenData, data.seattle.gov, data.cityofnewyork.us, and many other
municipal/state deployments — that returns plain
:class:`pandas.DataFrame` objects so morie's MRM modules can ingest
Socrata feeds with one call.

Quick usage
-----------

  >>> from morie.ingest.chicago import fetch_crime
  >>> df = fetch_crime(year=2024, max_features=10_000)
  >>> df.shape
  (10000, K)

  >>> from morie.ingest.chicago import fetch_socrata
  >>> df = fetch_socrata(
  ...     "https://data.cityofnewyork.us/resource/uip8-fykc.json",
  ...     where="arrest_year=2023",
  ...     max_features=5_000,
  ... )

CLI
---

::

    morie ingest chicago --year 2024 --out chicago-crime-2024.csv

"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd

DEFAULT_USER_AGENT = "morie/0.8.0 (+https://github.com/hadesllm/morie)"
DEFAULT_TIMEOUT_SECONDS = 60.0

# Socrata caps a single SoDA response at 50,000 rows server-side; we use
# a conservative page size that's still well above the default of 1,000.
DEFAULT_PAGE_SIZE = 50_000

# Canonical Chicago open-data Socrata resource ids.  The crime feed
# below is the documented main dataset (verified 2026-05-13):
#   https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
# Other resources can be added here as morie's analysis suite grows.
RESOURCE_REGISTRY: dict[str, str] = {
    "crime": "https://data.cityofchicago.org/resource/ijzp-q8t2.json",
}

CRIME_RESOURCE = RESOURCE_REGISTRY["crime"]


class ChicagoError(RuntimeError):
    """A Chicago / Socrata endpoint returned an HTTP error or no rows."""


def _socrata_get(
    resource_url: str,
    *,
    where: str | None = None,
    select: str | None = None,
    order: str | None = None,
    limit: int = DEFAULT_PAGE_SIZE,
    offset: int = 0,
    app_token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Single Socrata SoQL GET against ``resource_url``."""
    params: dict[str, Any] = {"$limit": limit, "$offset": offset}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select
    if order:
        params["$order"] = order
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if app_token:
        headers["X-App-Token"] = app_token
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        r = c.get(resource_url, params=params)
        if r.status_code >= 400:
            raise ChicagoError(f"socrata GET -> HTTP {r.status_code}: {r.text[:200]}")
        payload = r.json()
        if isinstance(payload, dict) and "error" in payload:
            raise ChicagoError(f"socrata error: {payload}")
        if not isinstance(payload, list):
            raise ChicagoError(f"socrata GET: unexpected payload shape: {type(payload).__name__}")
        return payload


def fetch_socrata(
    resource_url: str,
    *,
    where: str | None = None,
    select: str | None = None,
    order: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    max_features: int | None = None,
    app_token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Fetch every row from a Socrata SoDA JSON endpoint.

    Pages transparently through ``$offset`` until either the server
    returns fewer rows than ``page_size`` (last page) or ``max_features``
    is reached.

    Parameters
    ----------
    resource_url : str
        Full Socrata resource URL ending in ``.json``, e.g.
        ``https://data.cityofchicago.org/resource/ijzp-q8t2.json``.
    where, select, order : str | None
        SoQL clauses.  ``where`` is the equivalent of SQL ``WHERE``;
        ``select`` lets you project columns; ``order`` controls
        ordering.  See https://dev.socrata.com/docs/queries/.
    page_size : int
        Rows per request; capped at 50,000 server-side.
    max_features : int | None
        Optional hard cap on total returned rows.
    app_token : str | None
        Optional Socrata application token.  Anonymous calls work but
        share a throttled rate-limit pool; tokens give per-app quotas.
    timeout : float
        Per-request timeout in seconds.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _socrata_get(
            resource_url,
            where=where,
            select=select,
            order=order,
            limit=page_size,
            offset=offset,
            app_token=app_token,
            timeout=timeout,
        )
        if not page:
            break
        rows.extend(page)
        if max_features is not None and len(rows) >= max_features:
            rows = rows[:max_features]
            break
        if len(page) < page_size:
            break
        offset += len(page)

    if not rows:
        raise ChicagoError(f"socrata returned zero rows: where={where!r}")
    return pd.DataFrame(rows)


def fetch_crime(
    *,
    year: int | None = None,
    where: str | None = None,
    max_features: int | None = None,
    app_token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Pull the City of Chicago "Crimes – 2001 to Present" feed.

    Returns a DataFrame with the documented Socrata schema (preserving
    the source's snake_case column names verbatim):

      id, case_number, date, block, iucr, primary_type, description,
      location_description, arrest, domestic, beat, district, ward,
      community_area, fbi_code, x_coordinate, y_coordinate, year,
      updated_on, latitude, longitude.

    Parameters
    ----------
    year : int | None
        If set, server-side filter to ``year = <year>``.
    where : str | None
        Raw SoQL ``$where`` clause (overrides ``year``).
    max_features : int | None
        Cap on total returned rows.
    app_token : str | None
        Optional Socrata X-App-Token for higher rate limits.
    timeout : float
        Per-request timeout in seconds.
    """
    clause = where
    if clause is None and year is not None:
        clause = f"year = {int(year)}"
    return fetch_socrata(
        CRIME_RESOURCE,
        where=clause,
        max_features=max_features,
        app_token=app_token,
        timeout=timeout,
    )


def discover_resources() -> pd.DataFrame:
    """Return the built-in Chicago resource registry as a DataFrame."""
    return pd.DataFrame(
        [{"name": k, "url": v} for k, v in RESOURCE_REGISTRY.items()]
    )


# ----------------------------------------------------------------------
# CLI handler — wired into morie/runner.py
#
# Kept self-contained so ``morie ingest chicago ...`` can be added to the
# main dispatcher without further plumbing.


def cli(args: list[str]) -> int:
    """Handle ``morie ingest chicago ...``.  Returns exit code."""
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="morie ingest chicago",
                                description="Pull a City of Chicago Socrata feed.")
    p.add_argument("--resource",
                   help=f"Built-in resource name: {sorted(RESOURCE_REGISTRY)}")
    p.add_argument("--url", help="Direct Socrata resource URL (overrides --resource)")
    p.add_argument("--year", type=int, help="Filter to a single year (uses SoQL 'year =')")
    p.add_argument("--where", help="Raw SoQL $where clause (overrides --year)")
    p.add_argument("--max", type=int, dest="max_features", help="Cap returned rows")
    p.add_argument("--token", help="Socrata X-App-Token for higher rate limits")
    p.add_argument("--out", type=Path, help="CSV output path (stdout if omitted)")
    p.add_argument("--list", action="store_true",
                   help="List the built-in resource names and exit")
    ns = p.parse_args(args)

    if ns.list:
        sys.stdout.write(discover_resources().to_csv(index=False))
        return 0

    url = ns.url
    if url is None and ns.resource:
        if ns.resource not in RESOURCE_REGISTRY:
            sys.stderr.write(f"unknown resource {ns.resource!r}; try --list\n")
            return 2
        url = RESOURCE_REGISTRY[ns.resource]
    if url is None:
        url = CRIME_RESOURCE  # default for the bare `morie ingest chicago` call

    where = ns.where or (f"year = {ns.year}" if ns.year else None)
    df = fetch_socrata(url, where=where, max_features=ns.max_features,
                       app_token=ns.token)
    if ns.out:
        df.to_csv(ns.out, index=False)
        sys.stderr.write(f"wrote {ns.out}  ({len(df):,} rows, {len(df.columns)} cols)\n")
    else:
        sys.stdout.write(df.to_csv(index=False))
    return 0
