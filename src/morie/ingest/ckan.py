"""CKAN open-data portal client.

CKAN (https://ckan.org) is the open-source data-portal stack used by
Canada (open.canada.ca), the United States (data.gov), the United
Kingdom (data.gov.uk), the European Union (data.europa.eu), and many
provincial / municipal portals.  Every CKAN endpoint exposes the
same Action API (`/api/3/action/<verb>`), so a single thin client
covers every portal a sociolegal researcher is likely to touch.

Quick usage
-----------

  >>> from morie.ingest.ckan import Client
  >>> c = Client("https://open.canada.ca/data")
  >>> meta = c.package_show("canadian-postsecondary-alcohol-and-drug-use-survey")
  >>> meta["title"]
  'Canadian Postsecondary Alcohol and Drug Use Survey (CPADS)'
  >>> resources = meta["resources"]
  >>> csv_url = next(r["url"] for r in resources if r["format"].lower() == "csv")
  >>> df = c.read_resource(csv_url)
  >>> df.shape
  (N, K)

A common idiom — fetch every CSV in a package as one DataFrame per
resource — is the :func:`fetch_package_csvs` helper:

  >>> dfs = fetch_package_csvs(
  ...     "https://open.canada.ca/data",
  ...     "canadian-postsecondary-alcohol-and-drug-use-survey",
  ... )
  >>> dfs.keys()
  dict_keys(['CPADS Alcohol.csv', 'CPADS Cannabis.csv', ...])

CLI
---

::

    morie ingest ckan --portal https://open.canada.ca/data \\
                      --package canadian-postsecondary-alcohol-and-drug-use-survey \\
                      --out /tmp/cpads/

"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx
import pandas as pd

DEFAULT_USER_AGENT = "morie/0.5.0 (+https://github.com/hadesllm/morie)"
DEFAULT_TIMEOUT_SECONDS = 30.0


class CKANError(RuntimeError):
    """A CKAN API call returned success=False or an HTTP error."""


@dataclass
class Client:
    """Thin CKAN Action-API client.

    Parameters
    ----------
    portal : str
        Base URL of the portal, with or without trailing slash.  The
        Action API is at ``<portal>/api/3/action/<verb>``.  Examples:
        ``https://open.canada.ca/data``, ``https://data.gov.uk``,
        ``https://data.ontario.ca``.
    timeout : float, default 30.0
        Per-request timeout in seconds.
    user_agent : str, default ``"morie/0.5.0"``
        Sent as the ``User-Agent`` header; some portals rate-limit
        anonymous clients.
    api_key : str | None, default None
        Optional CKAN API key (rare for open-data portals).
    """

    portal: str
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    user_agent: str = DEFAULT_USER_AGENT
    api_key: str | None = None
    _client: httpx.Client | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # context-manager + connection lifecycle

    def __post_init__(self) -> None:
        self.portal = self.portal.rstrip("/")
        headers: dict[str, str] = {"User-Agent": self.user_agent}
        if self.api_key:
            headers["Authorization"] = self.api_key
        self._client = httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:  # type: ignore[override]
        self.close()

    # ------------------------------------------------------------------
    # Low-level: a single Action-API call

    def _call(self, action: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.portal}/api/3/action/{action}"
        assert self._client is not None  # noqa: S101 — closed by __exit__
        r = self._client.get(url, params=params or {})
        if r.status_code >= 400:
            raise CKANError(f"{action} -> HTTP {r.status_code}: {r.text[:200]}")
        payload = r.json()
        if not payload.get("success", False):
            error = payload.get("error", {})
            raise CKANError(f"{action} failed: {error}")
        return payload["result"]

    # ------------------------------------------------------------------
    # High-level: the verbs sociolegal users actually need

    def package_search(self, query: str | None = None, *, rows: int = 100, start: int = 0,
                       fq: str | None = None) -> dict[str, Any]:
        """Search for packages (datasets) by free-text or filter query.

        Returns the raw CKAN response: ``{"count": N, "results": [...]}``.
        """
        params: dict[str, Any] = {"rows": rows, "start": start}
        if query:
            params["q"] = query
        if fq:
            params["fq"] = fq
        return self._call("package_search", params)

    def package_show(self, package_id: str) -> dict[str, Any]:
        """Get the metadata for a single package by id or slug."""
        return self._call("package_show", {"id": package_id})

    def resource_show(self, resource_id: str) -> dict[str, Any]:
        """Get the metadata for a single resource (file) by id."""
        return self._call("resource_show", {"id": resource_id})

    def read_resource(self, url_or_id: str, *, as_format: str | None = None) -> pd.DataFrame:
        """Fetch a resource as a pandas DataFrame.

        ``url_or_id`` may be the direct download URL of the resource
        (as it appears in ``resource["url"]``) or a CKAN resource id,
        in which case the URL is resolved via ``resource_show``.

        Format detection: if ``as_format`` is given, use it.  Otherwise,
        sniff from the URL extension (csv / tsv / xlsx / json /
        parquet); fall back to CSV on unknown extensions.
        """
        if not url_or_id.startswith(("http://", "https://")):
            meta = self.resource_show(url_or_id)
            url = meta["url"]
            fmt = (as_format or meta.get("format") or "csv").lower()
        else:
            url = url_or_id
            fmt = (as_format or url.rsplit(".", 1)[-1] or "csv").lower()

        assert self._client is not None  # noqa: S101
        r = self._client.get(url)
        if r.status_code >= 400:
            raise CKANError(f"resource fetch -> HTTP {r.status_code}: {url}")
        content = r.content

        if fmt in {"csv"}:
            return pd.read_csv(io.BytesIO(content))
        if fmt in {"tsv", "tab"}:
            return pd.read_csv(io.BytesIO(content), sep="\t")
        if fmt in {"xlsx", "xls"}:
            return pd.read_excel(io.BytesIO(content))
        if fmt in {"json"}:
            return pd.read_json(io.BytesIO(content))
        if fmt in {"parquet"}:
            return pd.read_parquet(io.BytesIO(content))
        # Default: try CSV — most open-data resources are CSV with
        # mis-labelled MIME types.
        return pd.read_csv(io.BytesIO(content))


# ----------------------------------------------------------------------
# One-shot helpers — the 80%-case API


def fetch_package_csvs(portal: str, package_id: str) -> dict[str, pd.DataFrame]:
    """Fetch every CSV (and TSV) resource of a CKAN package.

    Returns a dict keyed by resource ``name`` (or ``url`` if name is
    missing) -> DataFrame.

    Skips resources that aren't CSV/TSV; use :class:`Client` directly
    if you need the others.
    """
    with Client(portal) as c:
        pkg = c.package_show(package_id)
        out: dict[str, pd.DataFrame] = {}
        for r in pkg.get("resources", []):
            fmt = (r.get("format") or "").lower()
            if fmt not in {"csv", "tsv"}:
                continue
            key = r.get("name") or r.get("url") or r.get("id") or fmt
            try:
                out[key] = c.read_resource(r["url"], as_format=fmt)
            except (CKANError, Exception) as exc:  # noqa: BLE001 — best-effort
                # Skip individual failures; the overall fetch still returns the others.
                out[f"_failed_{key}"] = pd.DataFrame({"error": [str(exc)]})
        return out


def search_packages(portal: str, query: str, *, rows: int = 50) -> pd.DataFrame:
    """Search a portal for packages, return a DataFrame of metadata."""
    with Client(portal) as c:
        resp = c.package_search(query, rows=rows)
        results = resp.get("results", [])
        # Flatten the most-useful fields
        rows_out = [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "title": p.get("title"),
                "organization": (p.get("organization") or {}).get("title"),
                "license_id": p.get("license_id"),
                "metadata_modified": p.get("metadata_modified"),
                "num_resources": p.get("num_resources"),
                "url": urljoin(portal.rstrip("/") + "/", f"dataset/{p.get('name', '')}"),
            }
            for p in results
        ]
        return pd.DataFrame(rows_out)


# ----------------------------------------------------------------------
# CLI handler — wired into morie/runner.py


def cli(args: list[str]) -> int:
    """Handle ``morie ingest ckan ...``.  Returns exit code."""
    import argparse
    import json
    import sys
    from pathlib import Path

    p = argparse.ArgumentParser(prog="morie ingest ckan",
                                description="Pull a CKAN package or search results into local CSVs.")
    p.add_argument("--portal", required=True, help="Base URL of the CKAN portal, e.g. https://open.canada.ca/data")
    p.add_argument("--package", help="Package id / slug to download every CSV resource of")
    p.add_argument("--search", help="Free-text search; prints matching packages as JSON to stdout")
    p.add_argument("--rows", type=int, default=20, help="Max search results to return")
    p.add_argument("--out", type=Path, default=Path("./ckan-out"), help="Output directory for downloaded CSVs")
    ns = p.parse_args(args)

    if ns.search:
        df = search_packages(ns.portal, ns.search, rows=ns.rows)
        sys.stdout.write(df.to_csv(index=False))
        return 0

    if ns.package:
        ns.out.mkdir(parents=True, exist_ok=True)
        dfs = fetch_package_csvs(ns.portal, ns.package)
        for key, df in dfs.items():
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
            target = ns.out / f"{safe}.csv"
            df.to_csv(target, index=False)
            sys.stderr.write(f"wrote {target}  ({len(df):,} rows, {len(df.columns)} cols)\n")
        sys.stdout.write(json.dumps({"package": ns.package, "files": list(dfs)}, indent=2) + "\n")
        return 0

    p.error("provide --package or --search")
    return 2
