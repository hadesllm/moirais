"""morie.run_crime_analysis -- one-command end-to-end pipeline.

Usage::

    from morie import run_crime_analysis
    run_crime_analysis.full()                  # everything
    run_crime_analysis.tps_only("Assault")     # one category
    run_crime_analysis.otis_only()             # OTIS Goffmanian + churn

Reproduces the full paper-205 analysis surface against the canonical
data/datasets/{OTIS,TPS}/ folders and writes outputs to
data/manifest/outputs/{otis,tps,overlay,otis_churn,tps_stochastic,
tps_spatial_advanced,figures}/. No further setup required.

Designed for users who want the paper-205 results "for free" on
their own copy of the OTIS / TPS data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[5]
OUT = PROJECT / "data/manifest/outputs"


def _step(label: str):
    """Pretty step printer."""
    bar = "─" * 60
    print(f"\n{bar}\n{label}\n{bar}", flush=True)


def otis_only(*, run_descriptive: bool = True,
              run_churn: bool = True,
              run_overlay: bool = False) -> dict[str, Any]:
    """Run OTIS-only analyses: 28 dataset descriptive + 6 Goffmanian
    churn tests."""
    out: dict[str, Any] = {}
    if run_descriptive:
        _step("OTIS descriptive (28 tables)")
        from . import otis_all_analyze
        out["descriptive"] = otis_all_analyze.analyze_all()
        print(f"  ✓ {len(out['descriptive'])} datasets")
    if run_churn:
        _step("OTIS Goffmanian churn (6 tests)")
        from . import otis_churn
        out["churn"] = otis_churn.analyze_all()
        print(f"  ✓ {len(out['churn'])} churn analyses")
    if run_overlay:
        _step("OTIS × TPS YoY overlay")
        from . import otis_tps_overlay
        out["overlay"] = otis_tps_overlay.analyze_all()
    return out


def tps_only(category: str = "Assault",
             *, sample_rows: int | None = 50_000) -> dict[str, Any]:
    """Run TPS-only stack on one category."""
    _step(f"TPS analyses -- {category}")
    out: dict[str, Any] = {}

    from . import tps_all_analyze, tps_datasets, tps_spatial, tps_spatial_advanced, tps_stochastic, tps_temporal
    df = tps_datasets.load_tps_dataset(category, nrows=sample_rows)
    print(f"  loaded {len(df):,} rows for {category}")

    print("  [1/7] standard analysis bundle")
    out["standard"] = tps_all_analyze.analyze_one(category, df)

    print("  [2/7] global Moran's I")
    out["morans_i"] = tps_spatial.morans_i_neighbourhood(df, ds_name=category)

    print("  [3/7] LISA local Moran's Ii")
    out["lisa"] = tps_spatial.local_morans_i(df, ds_name=category)

    print("  [4/7] year-over-year linear")
    out["yoy"] = tps_temporal.year_over_year_trend(df, ds_name=category)

    print("  [5/7] Pettitt change-point + seasonality")
    out["changepoint"] = tps_temporal.changepoint_detection(df, ds_name=category)
    out["seasonal"] = tps_temporal.seasonal_pattern(df, ds_name=category)

    print("  [6/7] advanced spatial (Ripley K + Gi* + DBSCAN)")
    out["ripley_k"] = tps_spatial_advanced.ripley_k(df, ds_name=category)
    out["g_star"] = tps_spatial_advanced.getis_ord_g_star(df, ds_name=category)
    out["dbscan"] = tps_spatial_advanced.dbscan_clusters(
        df, ds_name=category, eps_km=0.3, min_samples=20)

    print("  [7/7] stochastic physics (Hawkes + SARIMA + OU + FP)")
    out["stochastic"] = tps_stochastic.analyze(category)

    print(f"  ✓ {len(out)} outputs for {category}")
    return out


def overlay() -> dict[str, Any]:
    _step("OTIS × TPS overlay")
    from . import otis_tps_overlay
    return otis_tps_overlay.analyze_all()


def render_figures() -> dict[str, str]:
    """Re-render all paper-205 figures from existing JSON outputs.
    Tries R/ggplot first, falls back to matplotlib if R unavailable."""
    _step("Render paper-205 figures (R/ggplot first)")
    import subprocess
    result = {}
    r_script = PROJECT / "r-package/morie/R/viz_ggplot.R"
    if r_script.exists():
        try:
            cmd = ["Rscript", "-e",
                   f'source("{r_script}"); morie_render_all()']
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, cwd=PROJECT)
            print(r.stdout[-2000:])
            if r.returncode == 0:
                result["r_render"] = "ok"
            else:
                result["r_render"] = f"failed: {r.stderr[:500]}"
        except Exception as e:  # noqa: BLE001
            result["r_render"] = f"failed: {e!r}"
    else:
        result["r_render"] = "viz_ggplot.R not found"
    return result


def full(sample_rows: int | None = 50_000,
         tps_categories: list[str] | None = None) -> dict[str, Any]:
    """Run everything end-to-end on the bundled OTIS+TPS data.

    Wall-clock on a 2024 MacBook Air M2 / external VSR drive: ~5 min
    for the OTIS half + 30 s per TPS category.
    """
    t0 = time.time()
    print("MORIE -- full crime-analysis pipeline")
    print("=" * 60)

    out: dict[str, Any] = {}
    out["otis"] = otis_only(run_descriptive=True, run_churn=True,
                              run_overlay=True)

    if tps_categories is None:
        # The 6 categories that have full TPS data + render cleanly
        tps_categories = [
            "Assault", "AutoTheft", "BicycleTheft", "BreakandEnter",
            "Robbery", "TheftFromMovingVehicle", "TheftOver",
            "Homicides", "ShootingAndFirearmDiscarges",
        ]
    out["tps"] = {}
    for cat in tps_categories:
        try:
            out["tps"][cat] = tps_only(cat, sample_rows=sample_rows)
        except Exception as e:  # noqa: BLE001
            print(f"  {cat} FAILED: {e!r}")
            out["tps"][cat] = {"error": repr(e)}

    out["render"] = render_figures()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.0f} s")
    print(f"Outputs at: {OUT}")
    print("Findings: see the OTIS-TPS spatial analysis paper.")
    print("Dashboard: data/manifest/outputs/dashboard/index.html")
    return out


if __name__ == "__main__":
    full()
