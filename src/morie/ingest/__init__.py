"""morie.ingest — adapters for open-data portals.

Currently shipped:
  - :mod:`morie.ingest.ckan`  — CKAN-portal package_search / package_show /
                                resource fetch, with auto-paging and
                                pandas-DataFrame return.
  - :mod:`morie.ingest.tps`   — Toronto Police Service open-data
                                JSON feeds.
  - :mod:`morie.ingest.siu`   — Special Investigations Unit director's
                                reports (PDF index + text extraction).

Each sub-module exposes:

  * A thin client object (``Client``) for advanced usage.
  * One-shot helpers that take a URL or dataset id and return a
    ``pandas.DataFrame``.
  * An optional CLI handler reachable as ``morie ingest <portal> ...``.

All sub-modules use :mod:`httpx` (already a hard morie dep) so users
need nothing additional to ingest a remote feed once morie is
installed.
"""

from __future__ import annotations

from . import ckan, siu, tps  # noqa: F401  re-exported for ergonomic access

__all__ = ["ckan", "tps", "siu"]
