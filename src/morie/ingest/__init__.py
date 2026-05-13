"""morie.ingest — adapters for open-data portals.

Currently shipped:
  - :mod:`morie.ingest.ckan`     — CKAN-portal package_search / package_show /
                                   resource fetch, with auto-paging and
                                   pandas-DataFrame return.
  - :mod:`morie.ingest.tps`      — Toronto Police Service open-data
                                   JSON feeds (ArcGIS Hub).
  - :mod:`morie.ingest.siu`      — Special Investigations Unit director's
                                   reports (PDF index + text extraction).
  - :mod:`morie.ingest.chicago`  — City of Chicago Socrata feeds, plus a
                                   generic :func:`fetch_socrata` helper
                                   that works against any Socrata portal
                                   (NYC OpenData, data.seattle.gov, …).
  - :mod:`morie.ingest.bigquery` — Google BigQuery public-data adapter
                                   (lazy import; needs the ``bigquery``
                                   extra).

Each sub-module exposes:

  * A thin client object (``Client``) for advanced usage.
  * One-shot helpers that take a URL or dataset id and return a
    ``pandas.DataFrame``.
  * An optional CLI handler reachable as ``morie ingest <portal> ...``.

All sub-modules use :mod:`httpx` (already a hard morie dep) so users
need nothing additional to ingest a remote feed once morie is
installed.  :mod:`morie.ingest.bigquery` is the one exception: it
lazy-imports ``google-cloud-bigquery`` and asks for the ``bigquery``
extra at first call.
"""

from __future__ import annotations

# NOTE: ``bigquery`` is intentionally NOT eager-imported here — the
# Google SDK is a heavy optional dependency and the submodule's
# top-level only imports lazy-friendly things (pandas + TYPE_CHECKING).
from . import chicago, ckan, siu, tps  # noqa: F401  re-exported for ergonomic access

__all__ = ["ckan", "tps", "siu", "chicago", "bigquery"]


def __getattr__(name: str):  # pragma: no cover — trivial PEP 562 stub
    # Use importlib.import_module rather than `from . import <name>`
    # because the latter routes back through this very __getattr__ and
    # infinite-recurses.  See PEP 562 "lazy submodule import" pattern.
    if name == "bigquery":
        import importlib
        return importlib.import_module(f"{__name__}.bigquery")
    raise AttributeError(f"module 'morie.ingest' has no attribute {name!r}")
