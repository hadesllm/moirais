"""moirais.otis_tps_overlay — cross-link OTIS (Ontario corrections) with
TPS (Toronto police) data.

Both feeds touch Toronto, so meaningful overlays:

1. **Year-over-year correlation** — does annual segregation/RC use in
   the OTIS "Toronto" region track annual TPS incident counts?
2. **Per-OTIS-region rollups** — total seg/RC counts per region, with
   the Toronto-region row joinable to TPS aggregates.
3. **Crime × confinement composite** — combine TPS composite-risk
   index (per-neighbourhood) with OTIS seg/RC totals (Toronto-region
   only) to surface high-correlation years.

All emit RichResult.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fn._richresult import RichResult


PROJECT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = PROJECT / "data/manifest/outputs/overlay"


def _safe_load_otis_b01() -> pd.DataFrame:
    from .otis_datasets import load_otis_dataset
    return load_otis_dataset("b01")


def _safe_load_otis_c01() -> pd.DataFrame:
    """c01 = total individuals in custody/RC/seg by gender × year."""
    from .otis_datasets import load_otis_dataset
    return load_otis_dataset("c01")


def _safe_load_tps(name: str, *, sample_rows: int | None = 50_000) -> pd.DataFrame:
    from .tps_datasets import load_tps_dataset
    return load_tps_dataset(name, nrows=sample_rows)


def _toronto_region_seg_by_year(df_b01: pd.DataFrame) -> pd.Series:
    """Total Toronto-region segregation placements per fiscal year."""
    df = df_b01.copy()
    if "Region_AtTimeOfPlacement" in df.columns:
        df = df[df["Region_AtTimeOfPlacement"] == "Toronto"]
    if "EndFiscalYear" not in df.columns:
        return pd.Series(dtype=int)
    s = df.groupby("EndFiscalYear").size()
    s.index = s.index.astype(int)
    return s


def _tps_incidents_by_year(df_tps: pd.DataFrame) -> pd.Series:
    yc = "OCC_YEAR" if "OCC_YEAR" in df_tps.columns else (
        "REPORT_YEAR" if "REPORT_YEAR" in df_tps.columns else None
    )
    if yc is None:
        return pd.Series(dtype=int)
    s = df_tps.groupby(yc).size()
    s = s[(s.index >= 1990) & (s.index <= 2030)]
    s.index = s.index.astype(int)
    return s


def yoy_correlation(tps_categories: list[str] | None = None,
                    *, sample_rows: int | None = 50_000) -> RichResult:
    """Year-by-year correlation between OTIS Toronto-region segregation
    placements and TPS crime incident counts (per category).
    """
    from .tps_datasets import TPS_REGISTRY
    cats = tps_categories or sorted(TPS_REGISTRY.keys())
    seg = _toronto_region_seg_by_year(_safe_load_otis_b01())
    if seg.size == 0:
        return RichResult(
            title="OTIS×TPS — YoY correlation",
            warnings=["OTIS b01 has no Toronto-region data"],
        )

    rows = []
    for cat in cats:
        try:
            tps = _tps_incidents_by_year(_safe_load_tps(cat, sample_rows=sample_rows))
            common = seg.index.intersection(tps.index)
            if common.size < 3:
                rows.append([cat, int(common.size), "n/a", "n/a", "n/a"])
                continue
            x = seg.loc[common].astype(float)
            y = tps.loc[common].astype(float)
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            # Approx p via Fisher z: only for small reporting
            n = common.size
            if np.isfinite(r) and n > 3 and abs(r) < 1:
                z = 0.5 * np.log((1 + r) / (1 - r))
                p = 2 * (1 - _norm_cdf(abs(z) * np.sqrt(n - 3)))
            else:
                p = float("nan")
            rows.append([cat, int(common.size), round(r, 4),
                         round(float(p), 6), f"{int(common.min())}–{int(common.max())}"])
        except Exception as e:
            rows.append([cat, "err", str(e)[:30], "n/a", "n/a"])

    rows.sort(key=lambda r: -abs(r[2]) if isinstance(r[2], float) else 0)
    return RichResult(
        title="OTIS×TPS — year-by-year correlation (Toronto region)",
        summary_lines=[
            ("OTIS source", "b01 — segregation placements (Toronto region)"),
            ("OTIS years",
                f"{int(seg.index.min())}–{int(seg.index.max())}"),
            ("TPS categories tested", len(cats)),
        ],
        tables=[{
            "title": "Pearson r between OTIS Toronto-region segregation count "
                     "and TPS incident count (year-aligned):",
            "headers": ["TPS category", "Common years", "Pearson r",
                        "p-value", "Year range"],
            "rows": rows,
        }],
        interpretation=(
            "Positive r = years with more Toronto-region segregation placements "
            "in OTIS coincide with more reported TPS incidents in that "
            "category. This is association, NOT causation. Toronto OTIS data "
            "covers only 2023–2025, so common-year samples are small."
        ),
    )


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def per_region_rollup(*, tps_total_by_year: pd.Series | None = None) -> RichResult:
    """OTIS seg/RC totals per region × year, with the Toronto row
    flagged for TPS-overlay use. Also splits by gender.
    """
    df = _safe_load_otis_b01()
    if "Region_AtTimeOfPlacement" not in df.columns:
        return RichResult(title="OTIS region rollup",
                          warnings=["region column missing"])
    by_region = (df.groupby(["EndFiscalYear", "Region_AtTimeOfPlacement"])
                   .size().unstack(fill_value=0).sort_index())
    return RichResult(
        title="OTIS — segregation placements per region × year",
        summary_lines=[
            ("Years", f"{int(by_region.index.min())}–"
                     f"{int(by_region.index.max())}"),
            ("Regions",
                ", ".join(map(str, by_region.columns))),
            ("Total placements",
                int(by_region.values.sum())),
            ("Toronto region total",
                int(by_region.get("Toronto", pd.Series(dtype=int)).sum())),
        ],
        tables=[{
            "title": "Counts by year × region:",
            "headers": ["Year"] + list(map(str, by_region.columns)),
            "rows": [[int(y)] + list(map(int, row))
                     for y, row in by_region.iterrows()],
        }],
        payload={"by_region": by_region.to_dict()},
    )


def composite_overlay(*, sample_rows: int | None = 30_000) -> RichResult:
    """For each TPS category, compute year-aligned correlation with
    OTIS Toronto-region segregation. Same idea as yoy_correlation but
    uses ALL fiscal years jointly via concat.
    """
    return yoy_correlation(sample_rows=sample_rows)


def analyze_all(out_dir: Path | None = None,
                *, sample_rows: int | None = 30_000) -> dict[str, RichResult]:
    out_dir = out_dir or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    for name, fn in [
        ("region_rollup", per_region_rollup),
        ("yoy_correlation",
            lambda: yoy_correlation(sample_rows=sample_rows)),
    ]:
        try:
            r = fn()
            results[name] = r
            (out_dir / f"overlay_{name}.txt").write_text(str(r))
            (out_dir / f"overlay_{name}.json").write_text(
                json.dumps(r.payload, indent=2, default=str,
                           ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            results[name] = RichResult(title=f"overlay {name} (failed)",
                                       warnings=[f"{type(e).__name__}: {e}"])
    return results
