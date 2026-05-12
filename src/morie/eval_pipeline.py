"""Higher-level eval pipeline -- end-to-end integration gates over
BigQuery-mirrored datasets, each pinned to a published ground-truth
claim about a headline coefficient or shape statistic.

Why this exists alongside ``morie.eval``:

    ``morie.eval`` (per-fn goldens) catches drift in *one function* --
    "did dnorm(0) change from 0.3989… last week?". That is correctness
    at the leaf.  But nothing there catches the kind of bug where a
    function change quietly breaks an *integration* -- e.g., a switch
    from HC1 to HC3 SEs in the OLS path that flips a published
    coefficient from significant to insignificant on a real-world
    dataset.

    ``morie.eval_pipeline`` catches *that*: each gate is a claim of
    the form "this analysis on this dataset should produce a headline
    coefficient in range X" -- sourced from a peer-reviewed paper or a
    government report. When a gate fails, the bug is somewhere in the
    chain (data loading, canonicalisation, the model fit, the SE
    formula); the eval points at the broken integration even when
    every per-fn unit test still passes.

The harness ships seed gates against well-known public datasets. New
gates land as ``DatasetGate(...)`` entries in ``SEED_GATES`` below,
or via ``morie.eval_pipeline.register(...)`` from anywhere
downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from . import bq

GateRunner = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class DatasetGate:
    """One end-to-end claim about an analysis result.

    name        : short ID, used in test parametrize and reports.
    slug        : database slug (under the configured remote endpoint
                  or matching a local SQLite mirror).
    description : the *claim* being verified, in plain English.
    citation    : URL or publication this claim is sourced from.
    runner      : callable(slug) -> dict of computed coefficients.
    expected    : per-key (lower, upper) tolerance interval. Failure
                  if the actual value falls outside the interval.
                  Use (None, threshold) for "must be <= threshold",
                  or (threshold, None) for "must be >= threshold".
    """

    name: str
    slug: str
    description: str
    citation: str
    runner: GateRunner
    expected: dict[str, tuple[float | None, float | None]]


@dataclass
class GateResult:
    gate: DatasetGate
    actual: dict[str, Any] | None
    status: str  # "pass" | "fail" | "skip" | "error"
    failures: list[str] = field(default_factory=list)
    detail: str = ""


def _in_band(value: float, lo: float | None, hi: float | None) -> bool:
    if value is None:
        return False
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def run_gate(g: DatasetGate) -> GateResult:
    """Execute one gate end-to-end.  Catches all exceptions so the
    suite reports per-gate, not a single hard failure."""
    try:
        actual = g.runner(g.slug)
    except (ConnectionError, OSError, TimeoutError) as e:
        return GateResult(gate=g, actual=None, status="skip",
                          detail=f"unreachable: {e!s}")
    except Exception as e:  # noqa: BLE001
        return GateResult(gate=g, actual=None, status="error",
                          detail=f"{type(e).__name__}: {e!s}")
    failures: list[str] = []
    for key, (lo, hi) in g.expected.items():
        v = actual.get(key)
        if not isinstance(v, (int, float)) or not _in_band(float(v), lo, hi):
            failures.append(
                f"{key}={v!r} expected in [{lo}, {hi}]"
            )
    status = "pass" if not failures else "fail"
    return GateResult(gate=g, actual=actual, status=status, failures=failures)


# ── runner helpers ───────────────────────────────────────────────────────────
# Tiny adaptors over morie.bq so each runner is one expression.

def _primary_table(slug: str) -> str:
    """Pick the first non-_meta table. Public mirrors typically have
    a ``_meta`` companion table for column descriptions; the actual
    data lives in a sibling table named after the slug."""
    tables = [t for t in bq.bq_tables(slug) if not t.startswith("_")]
    if not tables:
        raise ValueError(f"no user tables in {slug!r}")
    return tables[0]


def _row_count(slug: str) -> dict[str, Any]:
    """Row count via SELECT count(*) -- works against any SQL endpoint
    that allows simple SELECT."""
    table = _primary_table(slug)
    rows = bq.bq_query(slug, f'SELECT count(*) AS n FROM "{table}"')
    return {"row_count": float(rows[0]["n"])}


def _column_count(slug: str) -> dict[str, Any]:
    """Column count via a 1-row SELECT * -- read ``len()`` of the first
    returned row's keys."""
    table = _primary_table(slug)
    rows = bq.bq_query(slug, f'SELECT * FROM "{table}" LIMIT 1')
    if not rows:
        return {"column_count": 0.0}
    return {"column_count": float(len(rows[0]))}


# ── seed gates (the registry) ────────────────────────────────────────────────
# Conservative claims for v0 -- "this dataset has at least N rows and
# at least M columns" is enough to catch a regression where a mirror
# dropped tables or schemas changed shape.  Real coefficient gates
# land as the runners get richer.

SEED_GATES: list[DatasetGate] = [
    # Canonical reference dataset -- Fisher iris is 150 rows × 5 cols
    # everywhere it appears.  If this gate ever fails, something
    # corrupted the iris mirror (or the morie bq layer).
    DatasetGate(
        name="iris_classic_shape",
        slug="iris",
        description=(
            "The Fisher iris dataset has 150 rows and 5 columns "
            "(sepal_length, sepal_width, petal_length, petal_width, "
            "species).  Stable canonical dataset; any deviation means "
            "we corrupted the iris mirror."
        ),
        citation="Fisher (1936); R datasets::iris",
        runner=_row_count,
        expected={"row_count": (150.0, 150.0)},
    ),
    DatasetGate(
        name="iris_5_columns",
        slug="iris",
        description="Iris columns = exactly 5.",
        citation="Fisher (1936); R datasets::iris",
        runner=_column_count,
        expected={"column_count": (5.0, 5.0)},
    ),
    # Lower-bound integrity gates -- sampled mirrors of public
    # BigQuery datasets. We don't pin exact counts (they drift on
    # each refresh), but a sudden 10x drop signals a broken ingest.
    DatasetGate(
        name="austin_311_min_rows",
        slug="austin_311",
        description=(
            "Austin 311 service requests sample mirror should have "
            "at least 100k rows."
        ),
        citation="bigquery-public-data.austin_311.311_service_requests",
        runner=_row_count,
        expected={"row_count": (100_000.0, None)},
    ),
    DatasetGate(
        name="noaa_ghcn_min_rows",
        slug="noaa_ghcn",
        description=(
            "NOAA Global Historical Climatology Network observations "
            "mirror should have >=200k rows."
        ),
        citation="bigquery-public-data.noaa_ghcn_d.ghcnd_*",
        runner=_row_count,
        expected={"row_count": (200_000.0, None)},
    ),
    DatasetGate(
        name="worldbank_min_rows",
        slug="worldbank",
        description=(
            "World Bank indicators mirror should have >=10k rows."
        ),
        citation="bigquery-public-data.worldbank_wdi.indicators_data",
        runner=_row_count,
        expected={"row_count": (10_000.0, None)},
    ),
    DatasetGate(
        name="chicago_crime_min_rows",
        slug="chicago_crime",
        description=(
            "Chicago Police Department incidents mirror should have "
            ">=1M rows. Sudden drop signals a broken ingest."
        ),
        citation="bigquery-public-data.chicago_crime.crime",
        runner=_row_count,
        expected={"row_count": (1_000_000.0, None)},
    ),
]

_REGISTRY: list[DatasetGate] = list(SEED_GATES)


def register(gate: DatasetGate) -> None:
    """Append a gate to the registry. Useful for downstream test
    suites that want to extend coverage without editing this file."""
    _REGISTRY.append(gate)


def gates() -> list[DatasetGate]:
    """Snapshot of the current registry."""
    return list(_REGISTRY)


def run_all() -> list[GateResult]:
    return [run_gate(g) for g in _REGISTRY]


def summary(results: list[GateResult]) -> dict[str, int]:
    s = {"pass": 0, "fail": 0, "skip": 0, "error": 0, "total": len(results)}
    for r in results:
        s[r.status] = s.get(r.status, 0) + 1
    return s
