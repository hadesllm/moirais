"""`morie verify-earth-engine` -- one-command EE auth smoke check.

Companion to `morie verify-pollution` (Workstream 6). This CLI
confirms whether the Earth Engine auth pipeline is wired up end-to-
end before you commit to a real pipeline run.

Steps it attempts, in order:

1. Is the ``earthengine-api`` package installed?
2. Do we have credentials? (env vars or ADC)
3. Does ``ee.Initialize()`` succeed?
4. Can we issue a trivial query? (fetch one pixel of a small image)

Exit codes mirror ``verify-pollution``:
- 0  -- all stages passed
- 1  -- auth failure (missing credentials, expired token)
- 2  -- environment error (missing library, bad network)

See also ``.claude/plans/2026-04-17-pollution-health-mission.md`` §7
(post-registration smoke check) and
the project's internal Earth Engine registration notes.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def _result_row(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"stage": name, "ok": bool(ok), "detail": detail}


def _probe_library() -> tuple[bool, str]:
    try:
        import importlib.metadata as md

        import ee  # type: ignore  # noqa: F401
        try:
            ver = md.version("earthengine-api")
        except md.PackageNotFoundError:
            ver = "installed (version unknown)"
        return True, f"earthengine-api {ver}"
    except ImportError as exc:
        return False, f"ImportError: {exc}. Run: pip install earthengine-api"


def _probe_credentials() -> tuple[bool, str]:
    sa = os.environ.get("MORIE_EE_SERVICE_ACCOUNT")
    key = os.environ.get("MORIE_EE_KEY_PATH") or os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS"
    )
    project = os.environ.get("MORIE_EE_PROJECT") or os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )
    if sa and key and project:
        if not os.path.exists(key):
            return False, f"MORIE_EE_KEY_PATH points at missing file: {key}"
        return True, f"service account {sa} @ {project} (key at {key})"
    if project:
        return True, (
            f"user OAuth flow (project={project}; no SA configured -- "
            "will attempt interactive ADC)"
        )
    return False, (
        "no GCP project configured. Set MORIE_EE_PROJECT or "
        "GOOGLE_CLOUD_PROJECT. See howto/earth_engine_auth.md."
    )


def _probe_initialize() -> tuple[bool, str]:
    from morie.earth import MissingCredentialsError, _ensure_ee_initialized

    try:
        ee = _ensure_ee_initialized()
    except MissingCredentialsError as exc:
        return False, f"MissingCredentialsError: {exc}"
    except Exception as exc:   # pragma: no cover - surface the raw error
        return False, f"{type(exc).__name__}: {exc}"
    # ee.Initialize() succeeded -- return a hint about the auth mode.
    try:
        proj = getattr(ee.data, "_cloud_api_user_project", None)
        return True, f"ee.Initialize() succeeded (project={proj or '(unknown)'})"
    except Exception:
        return True, "ee.Initialize() succeeded"


def _probe_query() -> tuple[bool, str]:
    """Run a 1-pixel fetch to confirm quota + dataset access."""
    try:
        import ee  # type: ignore
        # Small, cheap image. Uses Sentinel-5P NO2 -- which the Toronto
        # MRP pipeline actually uses -- so this tests the relevant path.
        ic = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        img = (
            ic.filterDate("2024-01-01", "2024-01-15")
            .select("NO2_column_number_density")
            .mean()
        )
        pt = ee.Geometry.Point([-79.3832, 43.6532])  # Toronto
        val = (
            img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=pt, scale=1000
            )
            .getInfo()
        )
        if val:
            return True, f"fetched NO2 sample over Toronto -- {val}"
        return True, "fetched (empty sample -- cloud-cover window?); quota ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def handle_verify_earth_engine(args: argparse.Namespace) -> int:
    """Main CLI dispatch."""
    results: list[dict[str, Any]] = []

    ok, detail = _probe_library()
    results.append(_result_row("library", ok, detail))
    if not ok:
        _emit(results, as_json=args.json)
        return 2

    ok, detail = _probe_credentials()
    results.append(_result_row("credentials", ok, detail))
    if not ok:
        _emit(results, as_json=args.json)
        return 1

    ok, detail = _probe_initialize()
    results.append(_result_row("initialize", ok, detail))
    if not ok:
        _emit(results, as_json=args.json)
        return 1

    if args.skip_query:
        results.append(_result_row(
            "query", True, "skipped (--skip-query)",
        ))
        _emit(results, as_json=args.json)
        return 0

    ok, detail = _probe_query()
    results.append(_result_row("query", ok, detail))
    _emit(results, as_json=args.json)
    return 0 if ok else 1


def _emit(results: list[dict[str, Any]], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"stages": results}, indent=2))
        return

    print("=" * 66)
    print("  morie verify-earth-engine")
    print("=" * 66)
    for r in results:
        tick = "PASS" if r["ok"] else "FAIL"
        print(f"  [{tick}] {r['stage']:12s} -- {r['detail']}")

    overall = all(r["ok"] for r in results)
    print("-" * 66)
    print(f"STATUS: {'ok' if overall else 'failed'}")
    if not overall:
        failed_names = [r["stage"] for r in results if not r["ok"]]
        print(f"Failed stages: {', '.join(failed_names)}")
        print("Next steps: see the project documentation for Earth Engine auth setup.")


def register_subparser(subparsers) -> None:
    """Wire the subcommand into morie.runner's argparse tree."""
    p = subparsers.add_parser(
        "verify-earth-engine",
        help="Smoke-check Earth Engine auth (credentials + init + 1-pixel query).",
    )
    p.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip the live 1-pixel fetch stage (offline / quota-sensitive).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
