"""BigQuery-derived dataset analysis over a local SQLite catalog.

This module gives the moirais analytics surface a typed entry point
into a SQLite-mirrored slice of public BigQuery datasets — useful for
local, network-free exploration and CI work.

Two access paths:

    1. Local — point at the on-disk SQLite file directly. Fast, no
       network. Works against the bundled ``moirais_datasets.db`` plus
       any user-supplied SQLite mirrors of public BigQuery slices.

    2. Remote — a SQL-over-HTTP endpoint configured via the
       ``MOIRAIS_REMOTE_URL`` env var. Useful for exploration without
       copying GBs of SQLite locally.

The high-level helpers (``bq_describe``, ``bq_columns``, ``bq_summary``,
``bq_correlate``) auto-pick local first, fall through to remote when
configured.
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional remote SQL-over-HTTP endpoint. Unset by default — set
# ``MOIRAIS_REMOTE_URL`` to a base URL (e.g. ``https://example.org/data``)
# to enable the remote fallback path.
REMOTE_BASE_URL = os.environ.get("MOIRAIS_REMOTE_URL", "")

# Where any user-supplied SQLite mirrors live locally.
LOCAL_DB_DIR = Path(
    os.environ.get("MOIRAIS_LOCAL_DB_DIR", str(Path.home() / "moirais_data"))
)


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------


@dataclass
class TableInfo:
    """High-level summary of one table inside a BigQuery-mirror database."""

    db: str
    table: str
    row_count: int
    columns: list[dict[str, str]]  # [{"name": "...", "type": "..."}]
    sample: list[dict[str, Any]]   # first 5 rows, dict-shaped


@dataclass
class ColumnSummary:
    """Univariate descriptive stats for one numeric column."""

    db: str
    table: str
    column: str
    n: int
    n_missing: int
    mean: float | None
    median: float | None
    std: float | None
    min: float | None
    max: float | None


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _local_path(db: str) -> Path | None:
    """Return the path to a local copy of `db.sqlite` if it exists."""
    for candidate in (
        LOCAL_DB_DIR / f"{db}.sqlite",
        LOCAL_DB_DIR / f"{db}.db",
    ):
        if candidate.exists():
            return candidate
    return None


def _query_local(db: str, sql: str) -> list[dict[str, Any]]:
    path = _local_path(db)
    if path is None:
        raise FileNotFoundError(
            f"no local copy of {db!r}; set MOIRAIS_LOCAL_DB_DIR or use bq_query_remote"
        )
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        return [dict(r) for r in cur.fetchall()]


def _query_remote(db: str, sql: str) -> list[dict[str, Any]]:
    if not REMOTE_BASE_URL:
        # Use ConnectionError so callers that catch network-class
        # exceptions (e.g. eval_pipeline gate runners) treat "no
        # endpoint configured" as a skip rather than a hard error.
        raise ConnectionError(
            "no remote endpoint configured; set MOIRAIS_REMOTE_URL "
            "or supply a local SQLite mirror via MOIRAIS_LOCAL_DB_DIR"
        )
    qs = urllib.parse.urlencode({"sql": sql, "_shape": "objects"})
    url = f"{REMOTE_BASE_URL}/{db}.json?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "moirais.bq/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("rows", [])


def _query(db: str, sql: str) -> list[dict[str, Any]]:
    """Try local SQLite, fall through to the configured remote endpoint."""
    if _local_path(db) is not None:
        return _query_local(db, sql)
    return _query_remote(db, sql)


# --------------------------------------------------------------------------
# Public API — high-level helpers a moirais user calls directly
# --------------------------------------------------------------------------


def bq_query(db: str, sql: str) -> list[dict[str, Any]]:
    """Run an arbitrary SQL query against the BigQuery-mirror database ``db``.

    Auto-picks local SQLite if available; falls through to the
    configured remote endpoint otherwise. Returns a list of row dicts.

    Examples
    --------
    >>> rows = bq_query("nyc_311_historic", "SELECT borough, count(*) FROM nyc_311_historic GROUP BY 1")
    """
    return _query(db, sql)


def bq_tables(db: str) -> list[str]:
    """List every table in the database."""
    rows = _query(
        db,
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY 1",
    )
    return [r["name"] for r in rows]


def bq_columns(db: str, table: str | None = None) -> list[dict[str, str]]:
    """Schema for `table` (defaults to the first user table in the db)."""
    if table is None:
        tables = bq_tables(db)
        if not tables:
            return []
        table = tables[0]
    rows = _query(db, f"PRAGMA table_info({_quote_ident(table)})")
    return [{"name": r["name"], "type": (r.get("type") or "").upper()} for r in rows]


def bq_describe(db: str, table: str | None = None) -> TableInfo:
    """High-level summary: row count, schema, first 5 rows."""
    if table is None:
        tables = bq_tables(db)
        if not tables:
            raise ValueError(f"no tables in {db!r}")
        table = tables[0]
    cols = bq_columns(db, table)
    n_rows = _query(db, f"SELECT count(*) AS n FROM {_quote_ident(table)}")[0]["n"]
    sample = _query(db, f"SELECT * FROM {_quote_ident(table)} LIMIT 5")
    return TableInfo(db=db, table=table, row_count=int(n_rows), columns=cols, sample=sample)


def bq_summary(db: str, table: str, column: str) -> ColumnSummary:
    """Univariate stats for a numeric column. NaN-safe."""
    t, c = _quote_ident(table), _quote_ident(column)
    sql = f"""
        SELECT
            count(*)                            AS n,
            sum(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS n_missing,
            avg(CAST({c} AS REAL))              AS mean,
            min(CAST({c} AS REAL))              AS min_,
            max(CAST({c} AS REAL))              AS max_
        FROM {t}
        WHERE {c} IS NOT NULL
    """
    row = _query(db, sql)[0]
    # SQLite has no median or stddev built-in for older versions — compute
    # client-side from a SELECT of the column. For huge tables the user
    # should use a sample.
    vals = [r[column] for r in _query(db, f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT 100000")]
    nums = [float(v) for v in vals if isinstance(v, (int, float))]
    median = None
    std = None
    if nums:
        nums.sort()
        mid = len(nums) // 2
        median = nums[mid] if len(nums) % 2 else (nums[mid - 1] + nums[mid]) / 2
        m = sum(nums) / len(nums)
        std = (sum((x - m) ** 2 for x in nums) / max(1, len(nums) - 1)) ** 0.5
    return ColumnSummary(
        db=db,
        table=table,
        column=column,
        n=int(row["n"] or 0),
        n_missing=int(row["n_missing"] or 0),
        mean=row["mean"],
        median=median,
        std=std,
        min=row["min_"],
        max=row["max_"],
    )


def bq_correlate(db: str, table: str, x: str, y: str) -> float | None:
    """Pearson correlation between two numeric columns (NaN-safe)."""
    t, xc, yc = _quote_ident(table), _quote_ident(x), _quote_ident(y)
    rows = _query(
        db,
        f"SELECT CAST({xc} AS REAL) AS xv, CAST({yc} AS REAL) AS yv "
        f"FROM {t} WHERE {xc} IS NOT NULL AND {yc} IS NOT NULL LIMIT 200000",
    )
    pairs = [(r["xv"], r["yv"]) for r in rows if r["xv"] is not None and r["yv"] is not None]
    if len(pairs) < 2:
        return None
    n = len(pairs)
    sx = sum(p[0] for p in pairs)
    sy = sum(p[1] for p in pairs)
    sxx = sum(p[0] * p[0] for p in pairs)
    syy = sum(p[1] * p[1] for p in pairs)
    sxy = sum(p[0] * p[1] for p in pairs)
    num = n * sxy - sx * sy
    den_sq = (n * sxx - sx * sx) * (n * syy - sy * sy)
    if den_sq <= 0:
        return None
    return num / (den_sq**0.5)


# --------------------------------------------------------------------------
# SQL identifier safety
# --------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier defensively. SQLite accepts double-quotes."""
    if not name or '"' in name:
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return f'"{name}"'


__all__ = [
    "TableInfo",
    "ColumnSummary",
    "REMOTE_BASE_URL",
    "LOCAL_DB_DIR",
    "bq_query",
    "bq_tables",
    "bq_columns",
    "bq_describe",
    "bq_summary",
    "bq_correlate",
]
