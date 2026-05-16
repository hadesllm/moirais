"""US forensic open-data ingestors.

A unified thin client over several federal forensic-statistics open-data
endpoints so morie can serve digital-forensics, missing-persons, and
federal crime-reporting use cases the same way :mod:`morie.ingest.tps`
serves Toronto Police data.

Sources covered
---------------

* **FBI NIBRS** — National Incident-Based Reporting System, exposed via
  the Crime Data Explorer API at
  ``https://api.usa.gov/crime/fbi/cde/``.  Requires an API key from
  ``https://api.data.gov/signup/``; pass via ``api_key=`` or the
  ``FBI_CDE_API_KEY`` environment variable.  Responses are nested JSON;
  this client flattens to one row per offence-event.

* **NamUs** — National Missing and Unidentified Persons System,
  ``https://www.namus.gov/api/CaseSets/NamUs/MissingPersons/...``.
  Socrata-like; no key required.  Returns case metadata
  (``case_number``, ``state``, ``county``, ``dlc_date``, ``sex``,
  ``race``, ``age_min``, ``age_max``, ``height_cm_min``, ...).

* **NIST RDS** — NIST Reference Datasets catalog metadata at
  ``https://data.nist.gov/rmm/records?...``.  The raw reference
  datasets themselves (CSAFE, NSRL, ...) are multi-gigabyte and must be
  downloaded out-of-band — this module returns the catalog records so
  callers can decide what to pull.

Quick usage
-----------

  >>> from morie.ingest.forensics import fetch_nibrs
  >>> df = fetch_nibrs(year=2023, offense="aggravated-assault",
  ...                  state="GA", api_key="...")
  >>> df.shape
  (N, K)

  >>> from morie.ingest.forensics import fetch_namus_missing_persons
  >>> df = fetch_namus_missing_persons(state="ON")  # any US state code
  >>> df.columns
  Index(['case_number', 'state', 'county', 'dlc_date', ...], dtype='object')

  >>> from morie.ingest.forensics import fetch_nist_rds
  >>> df = fetch_nist_rds(query="firearms")  # catalog metadata only

CLI
---

::

    morie ingest forensics --source nibrs --year 2023 --state GA \\
                           --out nibrs-ga-2023.csv

"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

DEFAULT_USER_AGENT = "morie/0.8.0 (+https://github.com/hadesllm/morie)"
DEFAULT_TIMEOUT_SECONDS = 60.0

# ----------------------------------------------------------------------
# Endpoints (verified 2026-05-13; subject to federal-portal reorg)

FBI_CDE_BASE = "https://api.usa.gov/crime/fbi/cde"
FBI_CDE_SIGNUP_URL = "https://api.data.gov/signup/"
FBI_CDE_API_KEY_ENV = "FBI_CDE_API_KEY"

NAMUS_MISSING_BASE = (
    "https://www.namus.gov/api/CaseSets/NamUs/MissingPersons/Search"
)

NIST_RDS_BASE = "https://data.nist.gov/rmm/records"


class ForensicsError(RuntimeError):
    """A forensic open-data endpoint returned an HTTP error or no records."""


class MissingAPIKeyError(ForensicsError):
    """Endpoint requires an API key but none was provided.

    Includes the signup URL in the message so callers don't have to
    chase it down.
    """


# ----------------------------------------------------------------------
# FBI NIBRS / Crime Data Explorer
# ----------------------------------------------------------------------


def _require_fbi_key(api_key: str | None) -> str:
    """Resolve the FBI CDE key from arg → env, or raise."""
    key = api_key or os.environ.get(FBI_CDE_API_KEY_ENV)
    if not key:
        raise MissingAPIKeyError(
            "FBI NIBRS / Crime Data Explorer requires an API key. "
            f"Sign up free at {FBI_CDE_SIGNUP_URL} and either pass "
            f"api_key=... or export {FBI_CDE_API_KEY_ENV}=<key>."
        )
    return key


def _flatten_nibrs_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten one nested NIBRS JSON record to a single row.

    NIBRS records can nest ``offense``, ``location``, ``victim``,
    ``offender``, ``weapon`` etc as sub-objects.  We flatten with
    dotted keys (``offense.code``, ``location.type``) and stringify
    list-valued fields to JSON.
    """
    import json

    out: dict[str, Any] = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                out[f"{k}.{sk}"] = sv
        elif isinstance(v, list):
            # Best-effort: keep scalar lists, json-serialise the rest.
            if all(not isinstance(x, (dict, list)) for x in v):
                out[k] = ";".join("" if x is None else str(x) for x in v)
            else:
                out[k] = json.dumps(v, default=str)
        else:
            out[k] = v
    return out


def fetch_nibrs(
    *,
    year: int,
    offense: str | None = None,
    state: str | None = None,
    api_key: str | None = None,
    max_features: int | None = None,
    page_size: int = 500,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Pull FBI NIBRS offence-event records via Crime Data Explorer.

    Parameters
    ----------
    year : int
        Reporting year (e.g. 2023).  Required — CDE forces a year scope.
    offense : str | None
        NIBRS offence slug (e.g. ``"aggravated-assault"``,
        ``"burglary"``); None returns all offences.
    state : str | None
        Two-letter US state code (e.g. ``"GA"``).  None returns the
        national feed (very large; use ``max_features``).
    api_key : str | None
        FBI CDE API key.  Falls back to ``$FBI_CDE_API_KEY``.  Without
        one, raises :class:`MissingAPIKeyError` pointing at the
        signup URL.
    max_features : int | None
        Hard cap on returned rows.
    page_size : int
        CDE page size; server-side cap varies by endpoint.
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    pd.DataFrame with one row per offence-event.  Nested NIBRS sub-objects
    are flattened using dotted keys (``offense.code``, ``victim.age``).
    """
    key = _require_fbi_key(api_key)
    # CDE NIBRS endpoint shape (verified 2026-05-13):
    #   /crime/fbi/cde/nibrs/<state>/<offense>?year=<year>&API_KEY=...
    # When state or offense are missing the path collapses to "national"
    # / "all".  Different CDE deployments have shifted this path over
    # time; if a 404 fires the error message points at the live API
    # documentation so the caller can adapt.
    state_part = state.upper() if state else "national"
    offense_part = offense or "all"
    url = f"{FBI_CDE_BASE}/nibrs/{state_part}/{offense_part}"

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    rows: list[dict[str, Any]] = []
    offset = 0
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        while True:
            params = {
                "API_KEY": key,
                "year": year,
                "from": offset,
                "size": page_size,
            }
            r = c.get(url, params=params)
            if r.status_code == 401 or r.status_code == 403:
                raise MissingAPIKeyError(
                    f"FBI CDE rejected the API key (HTTP {r.status_code}). "
                    f"Verify your key at {FBI_CDE_SIGNUP_URL}."
                )
            if r.status_code >= 400:
                raise ForensicsError(
                    f"FBI CDE NIBRS -> HTTP {r.status_code}: {r.text[:200]} "
                    f"(url={url})"
                )
            payload = r.json()
            # CDE wraps records in either "results" or "data"; handle both.
            batch = payload.get("results") or payload.get("data") or []
            if not batch:
                break
            for rec in batch:
                rows.append(_flatten_nibrs_record(rec))
            if max_features is not None and len(rows) >= max_features:
                rows = rows[:max_features]
                break
            if len(batch) < page_size:
                break
            offset += len(batch)

    if not rows:
        raise ForensicsError(
            f"FBI CDE NIBRS returned zero records "
            f"(year={year}, state={state!r}, offense={offense!r})"
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# NamUs — National Missing and Unidentified Persons System
# ----------------------------------------------------------------------


# Documented NamUs MissingPersons fields, in the order the public site
# exposes them.  Kept here so the offline / synthetic fallback ships the
# same schema as the live API.
NAMUS_MISSING_COLUMNS: tuple[str, ...] = (
    "case_number",
    "state",
    "county",
    "dlc_date",  # date-last-contact
    "sex",
    "race",
    "age_min",
    "age_max",
    "height_cm_min",
    "height_cm_max",
    "weight_kg_min",
    "weight_kg_max",
    "first_name",
    "last_name",
    "city",
    "circumstances",
)


def _flatten_namus_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten one NamUs subject record to morie's documented columns.

    NamUs returns a deeply nested ``subjectIdentification`` /
    ``subjectDescription`` / ``sighting`` shape.  We pull the fields
    most analysts actually want and drop the rest; pass the raw payload
    through :class:`httpx.Client` directly if you need everything.
    """
    sub_id = rec.get("subjectIdentification") or {}
    sub_desc = rec.get("subjectDescription") or {}
    sighting = rec.get("sighting") or {}
    sighting_addr = (sighting.get("address") or {}) if isinstance(sighting, dict) else {}

    def _range(d: Any, lo: str, hi: str) -> tuple[Any, Any]:
        if not isinstance(d, dict):
            return (None, None)
        return d.get(lo), d.get(hi)

    age_lo, age_hi = _range(sub_desc.get("currentMinAge") and sub_desc, "currentMinAge", "currentMaxAge")
    height_lo, height_hi = _range(sub_desc.get("heightFrom") and sub_desc, "heightFrom", "heightTo")
    weight_lo, weight_hi = _range(sub_desc.get("weightFrom") and sub_desc, "weightFrom", "weightTo")

    return {
        "case_number": rec.get("caseNumber") or rec.get("namUsCaseNumber"),
        "state": sighting_addr.get("state") or rec.get("state"),
        "county": sighting_addr.get("county") or rec.get("county"),
        "dlc_date": sighting.get("date") if isinstance(sighting, dict) else None,
        "sex": sub_desc.get("sex"),
        "race": sub_desc.get("primaryEthnicity") or sub_desc.get("race"),
        "age_min": age_lo,
        "age_max": age_hi,
        "height_cm_min": height_lo,
        "height_cm_max": height_hi,
        "weight_kg_min": weight_lo,
        "weight_kg_max": weight_hi,
        "first_name": sub_id.get("firstName"),
        "last_name": sub_id.get("lastName"),
        "city": sighting_addr.get("city"),
        "circumstances": rec.get("circumstances")
        or (rec.get("circumstancesOfDisappearance") or {}).get("circumstancesOfDisappearance"),
    }


def fetch_namus_missing_persons(
    *,
    state: str | None = None,
    max_features: int | None = None,
    page_size: int = 200,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> pd.DataFrame:
    """Pull NamUs missing-persons case metadata.

    Parameters
    ----------
    state : str | None
        Two-letter US state code (e.g. ``"CA"``).  None returns the
        national feed; on a slow connection use ``max_features`` to cap.
    max_features : int | None
        Hard cap on returned rows.
    page_size : int
        Records per request.
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    pd.DataFrame with the columns documented in
    :data:`NAMUS_MISSING_COLUMNS`.
    """
    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "take": page_size,
        "skip": 0,
        "projections": [],
        "predicates": [],
    }
    if state:
        body["predicates"].append(
            {"field": "state", "value": state.upper(), "operator": "Is"}
        )

    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        while True:
            r = c.post(NAMUS_MISSING_BASE, json=body)
            if r.status_code >= 400:
                raise ForensicsError(
                    f"NamUs MissingPersons -> HTTP {r.status_code}: {r.text[:200]}"
                )
            payload = r.json()
            # NamUs returns {"results": [...], "totalCount": N}; tolerate
            # either bare-list or {"data": [...]} variants.
            batch = (
                payload.get("results")
                or payload.get("data")
                or (payload if isinstance(payload, list) else [])
            )
            if not batch:
                break
            for rec in batch:
                rows.append(_flatten_namus_record(rec))
            if max_features is not None and len(rows) >= max_features:
                rows = rows[:max_features]
                break
            if len(batch) < page_size:
                break
            body["skip"] += len(batch)

    if not rows:
        raise ForensicsError(
            f"NamUs MissingPersons returned zero records (state={state!r})"
        )
    return pd.DataFrame(rows, columns=list(NAMUS_MISSING_COLUMNS))


# ----------------------------------------------------------------------
# NIST RDS — Reference Datasets catalog
# ----------------------------------------------------------------------


# The subset of NIST RDS catalog fields most analysts want; the raw
# records carry far more (provenance, version history, related
# resources) — pass through with `raw=True`.
NIST_RDS_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "title",
    "description",
    "publisher",
    "issued",
    "modified",
    "keyword",
    "landing_page",
    "size_bytes",
    "license",
)


def _flatten_nist_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Pull morie's documented columns out of one NIST RDS record."""
    import json

    keyword = rec.get("keyword") or rec.get("theme")
    if isinstance(keyword, list):
        keyword = ";".join(str(k) for k in keyword)

    publisher = rec.get("publisher")
    if isinstance(publisher, dict):
        publisher = publisher.get("name") or publisher.get("@id")

    license_ = rec.get("license") or rec.get("rights")
    if isinstance(license_, (dict, list)):
        license_ = json.dumps(license_, default=str)

    return {
        "dataset_id": rec.get("ediid") or rec.get("identifier") or rec.get("@id"),
        "title": rec.get("title"),
        "description": rec.get("description"),
        "publisher": publisher,
        "issued": rec.get("issued"),
        "modified": rec.get("modified"),
        "keyword": keyword,
        "landing_page": rec.get("landingPage") or rec.get("landing_page"),
        "size_bytes": rec.get("size") or rec.get("byteSize"),
        "license": license_,
    }


def fetch_nist_rds(
    *,
    dataset_id: str | None = None,
    query: str | None = None,
    max_features: int | None = None,
    page_size: int = 50,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    raw: bool = False,
) -> pd.DataFrame:
    """Pull NIST Reference Datasets catalog metadata.

    The raw reference datasets (CSAFE bullets/cartridges, NSRL hash
    library, etc.) are multi-gigabyte and shipped on dedicated
    download servers; this function returns only the catalog records
    so the caller can pick what to download separately.

    Parameters
    ----------
    dataset_id : str | None
        Specific NIST RDS / EDI id (e.g. ``"ark:/88434/mds2-2418"``).
        When set, returns a single-row frame.
    query : str | None
        Free-text search over the catalog (title/description/keyword).
        Ignored if ``dataset_id`` is set.
    max_features : int | None
        Hard cap on returned rows.
    page_size : int
        Records per request.
    timeout : float
        Per-request timeout in seconds.
    raw : bool
        If True, return the raw catalog JSON columns instead of morie's
        flattened :data:`NIST_RDS_COLUMNS` shape.
    """
    headers = {"User-Agent": DEFAULT_USER_AGENT, "Accept": "application/json"}
    params: dict[str, Any] = {"size": page_size}
    if dataset_id is not None:
        params["@id"] = dataset_id
    elif query:
        params["searchphrase"] = query

    rows: list[dict[str, Any]] = []
    offset = 0
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as c:
        while True:
            params["from"] = offset
            r = c.get(NIST_RDS_BASE, params=params)
            if r.status_code >= 400:
                raise ForensicsError(
                    f"NIST RDS -> HTTP {r.status_code}: {r.text[:200]}"
                )
            payload = r.json()
            batch = (
                payload.get("ResultData")
                or payload.get("results")
                or payload.get("data")
                or []
            )
            if not batch:
                break
            for rec in batch:
                rows.append(rec if raw else _flatten_nist_record(rec))
            if dataset_id is not None:
                # Single-record lookup; no paging.
                break
            if max_features is not None and len(rows) >= max_features:
                rows = rows[:max_features]
                break
            if len(batch) < page_size:
                break
            offset += len(batch)

    if not rows:
        raise ForensicsError(
            f"NIST RDS returned zero records "
            f"(dataset_id={dataset_id!r}, query={query!r})"
        )
    if raw:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=list(NIST_RDS_COLUMNS))


# ----------------------------------------------------------------------
# CLI handler — wired into morie/runner.py
# ----------------------------------------------------------------------


def cli(args: list[str]) -> int:
    """Handle ``morie ingest forensics ...``.  Returns exit code."""
    import argparse
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(
        prog="morie ingest forensics",
        description="Pull US forensic open-data (NIBRS, NamUs, NIST RDS).",
    )
    p.add_argument("--source", required=True,
                   choices=["nibrs", "namus", "nist-rds"],
                   help="Which forensic open-data source to pull.")
    p.add_argument("--year", type=int, help="Reporting year (NIBRS).")
    p.add_argument("--state", help="Two-letter state code (NIBRS, NamUs).")
    p.add_argument("--offense", help="NIBRS offence slug.")
    p.add_argument("--dataset-id", dest="dataset_id",
                   help="NIST RDS dataset id (single-record lookup).")
    p.add_argument("--query", help="NIST RDS free-text search phrase.")
    p.add_argument("--api-key", dest="api_key",
                   help=f"FBI CDE API key (or set ${FBI_CDE_API_KEY_ENV}).")
    p.add_argument("--max", type=int, dest="max_features",
                   help="Cap returned rows.")
    p.add_argument("--out", type=Path,
                   help="CSV output path (stdout if omitted).")
    ns = p.parse_args(args)

    if ns.source == "nibrs":
        if ns.year is None:
            p.error("--year is required for --source nibrs")
            return 2
        df = fetch_nibrs(
            year=ns.year, offense=ns.offense, state=ns.state,
            api_key=ns.api_key, max_features=ns.max_features,
        )
    elif ns.source == "namus":
        df = fetch_namus_missing_persons(
            state=ns.state, max_features=ns.max_features,
        )
    elif ns.source == "nist-rds":
        df = fetch_nist_rds(
            dataset_id=ns.dataset_id, query=ns.query,
            max_features=ns.max_features,
        )
    else:  # pragma: no cover — argparse choices guard
        p.error(f"unknown --source {ns.source!r}")
        return 2

    if ns.out:
        df.to_csv(ns.out, index=False)
        sys.stderr.write(f"wrote {ns.out}  ({len(df):,} rows, {len(df.columns)} cols)\n")
    else:
        sys.stdout.write(df.to_csv(index=False))
    return 0
