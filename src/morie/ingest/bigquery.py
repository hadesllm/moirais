"""Google BigQuery public-data adapter.

A very thin wrapper around :mod:`google.cloud.bigquery` that lets
morie pull BigQuery public datasets (e.g. ``bigquery-public-data.chicago_crime``,
``bigquery-public-data.new_york`` …) with a single call.

This module is **lazy-loaded**: ``google-cloud-bigquery`` is an *optional*
dependency, declared in the ``bigquery`` extra in :file:`pyproject.toml`.
The import is performed inside function bodies so that simply having
this file on disk does not impose the heavyweight Google SDK transitive
graph (auth, grpc, protobuf, …) on every morie user.

Authentication
--------------

Uses Application Default Credentials (ADC) — the same flow the rest of
the HADES-LLM Pi-rendered architecture relies on
(``project_hadesllm_pi_architecture``).  On a developer laptop that
means ``gcloud auth application-default login``; in a service context
it means a service-account key file or workload-identity binding,
picked up automatically via :func:`google.auth.default`.

Example
-------

  >>> from morie.ingest.bigquery import fetch_table
  >>> df = fetch_table(
  ...     project="bigquery-public-data",
  ...     dataset="chicago_crime",
  ...     table="crime",
  ...     where="year = 2024",
  ...     limit=10_000,
  ... )

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover — only for the type checker
    from google.cloud import bigquery as _bq_types  # noqa: F401


_BQ_IMPORT_HINT = (
    "morie.ingest.bigquery requires the optional google-cloud-bigquery "
    "dependency.  Install it with:\n\n"
    "    pip install 'morie[bigquery]'\n\n"
    "or directly:\n\n"
    "    pip install 'google-cloud-bigquery>=3.0'\n"
)


class BigQueryError(RuntimeError):
    """A BigQuery call failed or returned no rows."""


def _import_bq() -> Any:
    """Lazy-import :mod:`google.cloud.bigquery` with a clean error hint."""
    try:
        from google.cloud import bigquery as bq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — exercised by smoke tests
        raise ImportError(_BQ_IMPORT_HINT) from exc
    return bq


def _quote_ident(name: str) -> str:
    """Backtick-quote a BigQuery identifier, refusing anything obviously
    not a legal project/dataset/table name."""
    if not name or not all(c.isalnum() or c in {"_", "-"} for c in name):
        raise BigQueryError(f"illegal BigQuery identifier: {name!r}")
    return f"`{name}`"


def build_sql(
    project: str,
    dataset: str,
    table: str,
    *,
    where: str | None = None,
    limit: int | None = None,
    select: str = "*",
) -> str:
    """Build a parameter-safe ``SELECT`` against a fully-qualified table.

    Identifiers are validated and backtick-quoted; ``where`` is passed
    through unchanged (callers compose SoQL/SQL fragments themselves and
    are responsible for not injecting hostile clauses — same contract as
    :func:`morie.ingest.chicago.fetch_socrata`).
    """
    qproject = _quote_ident(project)
    qdataset = _quote_ident(dataset)
    qtable = _quote_ident(table)
    sql = f"SELECT {select} FROM {qproject}.{qdataset}.{qtable}"
    if where:
        sql += f"\nWHERE {where}"
    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"
    return sql


def get_client(
    *,
    project: str | None = None,
    credentials: Any | None = None,
) -> Any:
    """Return an authenticated :class:`google.cloud.bigquery.Client`.

    Uses :func:`google.auth.default` ADC discovery when ``credentials``
    is None — the same flow the HADES-LLM Pi uses.

    Parameters
    ----------
    project : str | None
        Billing project for the client.  ``None`` lets BigQuery infer
        from ADC; required when querying public datasets owned by
        another project (the *table's* project may differ from the
        *billing* project).
    credentials : google.auth.credentials.Credentials | None
        Override credentials.  ``None`` discovers via ADC.
    """
    bq = _import_bq()
    if credentials is None:
        try:
            from google.auth import default as _adc_default  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(_BQ_IMPORT_HINT) from exc
        credentials, adc_project = _adc_default()
        if project is None:
            project = adc_project
    return bq.Client(project=project, credentials=credentials)


def run_query(
    sql: str,
    *,
    project: str | None = None,
    client: Any | None = None,
    timeout: float | None = None,
) -> pd.DataFrame:
    """Execute ``sql`` and return the result as a DataFrame.

    Uses :meth:`google.cloud.bigquery.QueryJob.to_dataframe`, which
    pulls results via the BigQuery Storage API when ``pyarrow`` is
    available and falls back to the REST API otherwise.
    """
    c = client or get_client(project=project)
    job = c.query(sql)
    return job.result(timeout=timeout).to_dataframe()


def fetch_table(
    *,
    project: str,
    dataset: str,
    table: str,
    where: str | None = None,
    limit: int | None = None,
    select: str = "*",
    billing_project: str | None = None,
    client: Any | None = None,
    timeout: float | None = None,
) -> pd.DataFrame:
    """Pull a BigQuery table (or filtered slice) into a DataFrame.

    Parameters
    ----------
    project, dataset, table : str
        Fully-qualified table reference, e.g. project
        ``bigquery-public-data``, dataset ``chicago_crime``, table
        ``crime``.
    where : str | None
        Raw SQL ``WHERE`` clause (no leading ``WHERE``).
    limit : int | None
        Optional ``LIMIT``.
    select : str
        Projection list; defaults to ``*``.
    billing_project : str | None
        Project to bill the query to.  Public datasets cost the
        *caller's* project, not the dataset's owner.  ``None`` uses the
        ADC-discovered project.
    client : google.cloud.bigquery.Client | None
        Pre-built client; pass to reuse credentials across calls.
    timeout : float | None
        Total query timeout in seconds.
    """
    sql = build_sql(project, dataset, table,
                    where=where, limit=limit, select=select)
    return run_query(sql, project=billing_project, client=client, timeout=timeout)
