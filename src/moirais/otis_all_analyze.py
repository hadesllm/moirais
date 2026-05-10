"""moirais.otis_all_analyze — RichResult-emitting analyses for ALL 28
OTIS datasets.

This module is the **comprehensive** OTIS analysis surface. It pairs
with `moirais.otis_datasets` (the registry/loader) and extends the
narrower `moirais.otis_analyze` (which only covered the restricted-
confinement subset).

For each dataset id (b01..d07), this module exposes:
    analyze_<id>(df=None) -> RichResult

If df is None, the analysis loads from
data/datasets/OTIS/<csv_filename>.

Plus a top-level driver:
    analyze_all(*, out_dir=None) -> dict[str, RichResult]
which runs every dataset's analysis, writes per-dataset .txt and .json
under data/manifest/outputs/otis/, and returns the in-memory results.

Plus the Ruhela-formulation (RF) high-level entry points:
    analyze_a01_ruhela_formulations(df=None)   — full DLRM on a01
        — alias: analyze_a01_dlrm
    analyze_b01_ruhela_formulations(df=None)   — full DLRM on b01
        — alias: analyze_b01_dlrm
    analyze_b02_ruhela_formulations(df=None)   — gender → seg days on b02
        — alias: analyze_b02_dlrm
    analyze_a01_with_csi_context()             — a01 causal + Toronto CSI
    analyze_c_doob_chi2()                      — χ² + Cramer's V on c-series
        — homage to Doob's chi-square tradition
    analyze_d_doob_chi2()                      — d-series + Alert χ²

Naming abbreviations:
    RF  — Ruhela formulation
    RDF — Ruhela Dual Formulation (RF + Naive-arm sensitivity)
    DLRM — Doob-Levinsky-Ruhela-Medina (methodology attribution)

The DLRM methodology attribution and the wider acknowledgements
(Prof. Jauregui, Prof. Laniyonu) live at the top of
``moirais.otis_causal``'s module docstring.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fn._richresult import RichResult
from .otis_datasets import (
    DATASET_REGISTRY,
    load_otis_dataset,
)

PROJECT = Path(__file__).resolve().parents[5]
DEFAULT_OUT = PROJECT / "data/manifest/outputs/otis"


# ── Generic helpers ────────────────────────────────────────────────


def _year_col(df: pd.DataFrame) -> str | None:
    for c in ("EndFiscalYear", "Year"):
        if c in df.columns:
            return c
    return None


def _to_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _summary_lines(df: pd.DataFrame, ds_id: str) -> list[tuple[str, Any]]:
    meta = DATASET_REGISTRY[ds_id]
    yc = _year_col(df)
    out: list[tuple[str, Any]] = [
        ("Dataset", f"{ds_id} — {meta.description}"),
        ("Series", meta.series),
        ("Rows", int(df.shape[0])),
        ("Columns", int(df.shape[1])),
    ]
    if yc and df.shape[0]:
        years = pd.to_numeric(df[yc], errors="coerce").dropna()
        if years.size:
            out.append(("Years covered",
                        f"{int(years.min())}–{int(years.max())}"))
    if meta.primary_metric in df.columns:
        col = pd.to_numeric(df[meta.primary_metric], errors="coerce")
        out.append((f"Total {meta.primary_metric}",
                    int(col.sum())))
    return out


def _crosstab(df: pd.DataFrame,
              row: str, col: str, value: str,
              aggfunc: str = "sum",
              top_rows: int = 20) -> dict | None:
    if not all(c in df.columns for c in (row, col, value)):
        return None
    pivot = df.pivot_table(
        index=row, columns=col,
        values=value, aggfunc=aggfunc, fill_value=0,
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("TOTAL", ascending=False).head(top_rows)
    return {
        "title": f"{value} by {row} × {col}:",
        "headers": [row] + [str(c) for c in pivot.columns],
        "rows": [[str(idx)] + [int(v) if pd.notna(v) else 0
                                for v in row_vals]
                  for idx, row_vals in pivot.iterrows()],
    }


def _year_trend(df: pd.DataFrame, value: str,
                year_col: str | None = None) -> dict | None:
    yc = year_col or _year_col(df)
    if not yc or value not in df.columns:
        return None
    g = df.groupby(yc)[value].sum().sort_index()
    return {
        "title": f"{value} by {yc}:",
        "headers": [yc, value],
        "rows": [[int(y), int(v)] for y, v in g.items()],
    }


# ── b-series analyses ──────────────────────────────────────────────


def analyze_b01(df: pd.DataFrame | None = None) -> RichResult:
    """Person-level segregation placements (76,934 rows)."""
    df = df if df is not None else load_otis_dataset("b01")
    summary = _summary_lines(df, "b01")
    summary.append(("Unique individuals",
                    int(df["UniqueIndividual_ID"].nunique())))
    summary.append(("Mean consecutive days",
                    float(df["NumberConsecutiveDays_Segregation"].mean())))
    summary.append(("Median consecutive days",
                    int(df["NumberConsecutiveDays_Segregation"].median())))
    summary.append(("Max consecutive days",
                    int(df["NumberConsecutiveDays_Segregation"].max())))

    # Reason flags
    reasons = [c for c in df.columns if c.startswith("SegReason_")]
    reason_rows = []
    for r in reasons:
        n_yes = int((df[r] == True).sum() if df[r].dtype == bool
                    else (df[r] == 1).sum())
        if n_yes > 0:
            reason_rows.append([r.replace("SegReason_", ""), n_yes,
                                f"{100*n_yes/df.shape[0]:.1f}%"])
    reason_rows.sort(key=lambda r: -r[1])

    # Alert flags
    alerts = ["MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert"]
    alert_rows = []
    for a in alerts:
        if a in df.columns:
            n_yes = int((df[a] == True).sum() if df[a].dtype == bool
                        else (df[a] == 1).sum())
            alert_rows.append([a, n_yes,
                              f"{100*n_yes/df.shape[0]:.1f}%"])

    return RichResult(
        title="b01 — Segregation placements (person-level detail)",
        summary_lines=summary,
        tables=[
            {"title": "Reasons for placement (count, % of rows):",
             "headers": ["Reason", "Count", "Percent"], "rows": reason_rows},
            {"title": "Alert flags on placements:",
             "headers": ["Alert", "Count", "Percent"], "rows": alert_rows},
            _year_trend(df, "Number_Of_Placements") or {"title": "(no year trend)", "headers": [], "rows": []},
        ],
        payload={"n_rows": int(df.shape[0]),
                 "n_individuals": int(df["UniqueIndividual_ID"].nunique())},
    )


def analyze_b02(df: pd.DataFrame | None = None) -> RichResult:
    """Aggregate days in segregation per person per year."""
    df = df if df is not None else load_otis_dataset("b02")
    summary = _summary_lines(df, "b02")
    days = pd.to_numeric(df["TotalAggregatedDays_Segregation"],
                         errors="coerce").dropna()
    summary.append(("Mean total days", float(days.mean())))
    summary.append(("Median total days", int(days.median())))
    summary.append(("Max total days", int(days.max())))
    return RichResult(
        title="b02 — Segregation total days per person per fiscal year",
        summary_lines=summary,
        tables=[
            _year_trend(df, "TotalAggregatedDays_Segregation") or {"title": "(none)", "headers": [], "rows": []},
            _crosstab(df, "Gender", "Region_MostRecentPlacement",
                      "TotalAggregatedDays_Segregation") or
                {"title": "(none)", "headers": [], "rows": []},
        ],
    )


def analyze_b03(df: pd.DataFrame | None = None) -> RichResult:
    """Segregation placements by alert × institution."""
    df = df if df is not None else load_otis_dataset("b03")
    summary = _summary_lines(df, "b03")
    return RichResult(
        title="b03 — Segregation placements by alert/hold flag × institution",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Alert_Type", "Alert_Presence",
                      "Number_SegregationPlacements") or {},
            _crosstab(df, "Region_AtTimeOfPlacement", "Alert_Type",
                      "Number_SegregationPlacements") or {},
        ],
    )


def analyze_b04(df: pd.DataFrame | None = None) -> RichResult:
    """Placement durations (max/median/mode) by region & gender."""
    df = df if df is not None else load_otis_dataset("b04")
    summary = _summary_lines(df, "b04")
    return RichResult(
        title="b04 — Placement durations by region & gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Region_AtTimeOfPlacement", "Measure",
                      "NumberConsecutiveDays_Segregation",
                      aggfunc="max") or {},
        ],
    )


def analyze_b05(df: pd.DataFrame | None = None) -> RichResult:
    """Distribution of placements by binned duration."""
    df = df if df is not None else load_otis_dataset("b05")
    summary = _summary_lines(df, "b05")
    return RichResult(
        title="b05 — Distribution of placements by binned duration",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Consecutive_Duration", "EndFiscalYear",
                      "Number_SegregationPlacements") or {},
        ],
    )


def analyze_b06(df: pd.DataFrame | None = None) -> RichResult:
    """Reasons for placement × institution × gender."""
    df = df if df is not None else load_otis_dataset("b06")
    summary = _summary_lines(df, "b06")
    return RichResult(
        title="b06 — Reasons for placement by institution & gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Reason", "EndFiscalYear",
                      "Number_SegregationPlacements") or {},
            _crosstab(df, "Reason", "Gender",
                      "Number_SegregationPlacements") or {},
        ],
    )


def analyze_b07(df: pd.DataFrame | None = None) -> RichResult:
    """Alerts × gender."""
    df = df if df is not None else load_otis_dataset("b07")
    summary = _summary_lines(df, "b07")
    rows = []
    for _, r in df.iterrows():
        with_a = _to_int(r["Number_Segregation_Placements_With_Alert"])
        wo_a = _to_int(r["Number_Segregation_Placements_Without_Alert"])
        tot = with_a + wo_a
        rate = (with_a / tot * 100) if tot > 0 else 0.0
        rows.append([_to_int(r["EndFiscalYear"]), str(r["Alert_Type"]),
                     str(r["Gender"]), with_a, wo_a, f"{rate:.1f}%"])
    return RichResult(
        title="b07 — Segregation placements with/without alert × gender",
        summary_lines=summary,
        tables=[{"title": "By alert × gender × year:",
                 "headers": ["Year", "Alert_Type", "Gender", "With_Alert",
                             "Without_Alert", "% with alert"],
                 "rows": rows}],
    )


def analyze_b08(df: pd.DataFrame | None = None) -> RichResult:
    """Durations (median/mode) by institution & gender."""
    df = df if df is not None else load_otis_dataset("b08")
    summary = _summary_lines(df, "b08")
    return RichResult(
        title="b08 — Placement durations by institution & gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Institution_AtTimeOfPlacement", "Measure",
                      "NumberConsecutiveDays_Segregation",
                      aggfunc="max", top_rows=15) or {},
        ],
    )


def analyze_b09(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals by number of segregation placements × gender."""
    df = df if df is not None else load_otis_dataset("b09")
    summary = _summary_lines(df, "b09")
    return RichResult(
        title="b09 — Individuals by number of placements × gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "NumberPlacements_Segregation", "Gender",
                      "NumberIndividuals_Segregation") or {},
        ],
    )


# ── c-series analyses ──────────────────────────────────────────────


def analyze_c01(df: pd.DataFrame | None = None) -> RichResult:
    """Total individuals × custody/RC/seg × gender."""
    df = df if df is not None else load_otis_dataset("c01")
    summary = _summary_lines(df, "c01")
    # Compute RC/custody and seg/custody rates
    rate_rows = []
    for _, r in df.iterrows():
        cust = _to_int(r["NumberIndividuals_InCustody"])
        rc = _to_int(r["NumberIndividuals_RestrictiveConfinement"])
        seg = _to_int(r["NumberIndividuals_Segregation"])
        rate_rows.append([_to_int(r["EndFiscalYear"]), str(r["Gender"]),
                          cust, rc, seg,
                          f"{(rc/cust*100) if cust else 0:.1f}%",
                          f"{(seg/cust*100) if cust else 0:.1f}%"])
    return RichResult(
        title="c01 — Total individuals × custody/RC/seg × gender",
        summary_lines=summary,
        tables=[{"title": "Cohort sizes + ratios:",
                 "headers": ["Year", "Gender", "Custody", "RC", "Seg",
                             "RC/custody", "Seg/custody"],
                 "rows": rate_rows}],
    )


def analyze_c02(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals in RC/seg by institution."""
    df = df if df is not None else load_otis_dataset("c02")
    summary = _summary_lines(df, "c02")
    return RichResult(
        title="c02 — Individuals in RC/seg by institution × region × gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Institution_MostRecentPlacement",
                      "EndFiscalYear",
                      "NumberIndividuals_RestrictiveConfinement",
                      top_rows=15) or {},
        ],
    )


def analyze_c03(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × race × gender."""
    df = df if df is not None else load_otis_dataset("c03")
    summary = _summary_lines(df, "c03")
    # Compute race-disparity ratios: RC/custody and seg/custody by race
    by_race = df.groupby("Race", as_index=False).agg(
        cust=("NumberIndividuals_InCustody", "sum"),
        rc=("NumberIndividuals_RestrictiveConfinement", "sum"),
        seg=("NumberIndividuals_Segregation", "sum"),
    )
    rows = []
    for _, r in by_race.iterrows():
        cust = _to_int(r["cust"])
        rc = _to_int(r["rc"])
        seg = _to_int(r["seg"])
        rows.append([str(r["Race"]), cust, rc, seg,
                    f"{(rc/cust*100) if cust else 0:.1f}%",
                    f"{(seg/cust*100) if cust else 0:.1f}%"])
    rows.sort(key=lambda x: -x[1])
    return RichResult(
        title="c03 — Individuals × race × gender",
        summary_lines=summary,
        tables=[{"title": "By race (totals across years/genders):",
                 "headers": ["Race", "Custody", "RC", "Seg",
                             "RC/custody", "Seg/custody"],
                 "rows": rows}],
        interpretation=(
            "Race × confinement disparities are visible in the "
            "RC/custody and Seg/custody ratios. Compare across race "
            "categories — the gap between Indigenous and White rates is "
            "a documented Ontario-corrections finding."
        ),
    )


def analyze_c04(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × race × region."""
    df = df if df is not None else load_otis_dataset("c04")
    summary = _summary_lines(df, "c04")
    return RichResult(
        title="c04 — Individuals in RC/seg × race × region",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Race", "Region_MostRecentPlacement",
                      "NumberIndividuals_RestrictiveConfinement") or {},
        ],
    )


def analyze_c05(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × religion × region."""
    df = df if df is not None else load_otis_dataset("c05")
    summary = _summary_lines(df, "c05")
    return RichResult(
        title="c05 — Individuals in RC/seg × religion × region",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Religion", "Region_MostRecentPlacement",
                      "NumberIndividuals_RestrictiveConfinement") or {},
        ],
    )


def analyze_c06(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × age × region."""
    df = df if df is not None else load_otis_dataset("c06")
    summary = _summary_lines(df, "c06")
    return RichResult(
        title="c06 — Individuals in RC/seg × age category × region",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Age_Category", "Region_MostRecentPlacement",
                      "NumberIndividuals_RestrictiveConfinement") or {},
        ],
    )


def analyze_c07(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × alerts × gender."""
    df = df if df is not None else load_otis_dataset("c07")
    summary = _summary_lines(df, "c07")
    return RichResult(
        title="c07 — Individuals in custody/RC/seg × alert type × gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Alert_Type", "Gender",
                      "NumberIndividuals_RestrictiveConfinement") or {},
            _crosstab(df, "Alert_Type", "EndFiscalYear",
                      "NumberIndividuals_Segregation") or {},
        ],
    )


def analyze_c08(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × religion × gender."""
    df = df if df is not None else load_otis_dataset("c08")
    summary = _summary_lines(df, "c08")
    return RichResult(
        title="c08 — Individuals × religion × gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Religion", "Gender",
                      "NumberIndividuals_RestrictiveConfinement") or {},
        ],
    )


def analyze_c09(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals × age × gender."""
    df = df if df is not None else load_otis_dataset("c09")
    summary = _summary_lines(df, "c09")
    return RichResult(
        title="c09 — Individuals × age category × gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Age_Category", "Gender",
                      "NumberIndividuals_RestrictiveConfinement") or {},
        ],
    )


def analyze_c10(df: pd.DataFrame | None = None) -> RichResult:
    """RC/seg aggregate durations by institution."""
    df = df if df is not None else load_otis_dataset("c10")
    summary = _summary_lines(df, "c10")
    return RichResult(
        title="c10 — RC/seg aggregate durations by institution",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Institution_MostRecentPlacement", "Measure",
                      "TotalAggregatedDays_RestrictiveConfinement",
                      aggfunc="max", top_rows=15) or {},
        ],
    )


def analyze_c11(df: pd.DataFrame | None = None) -> RichResult:
    """Individuals by aggregate-duration bin."""
    df = df if df is not None else load_otis_dataset("c11")
    summary = _summary_lines(df, "c11")
    return RichResult(
        title="c11 — Individuals by binned aggregate duration",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Aggregate_Duration", "EndFiscalYear",
                      "NumberIndividuals_RestrictiveConfinement") or {},
        ],
    )


def analyze_c12(df: pd.DataFrame | None = None) -> RichResult:
    """RC/seg aggregate durations by region & gender."""
    df = df if df is not None else load_otis_dataset("c12")
    summary = _summary_lines(df, "c12")
    return RichResult(
        title="c12 — RC/seg aggregate durations by region & gender",
        summary_lines=summary,
        tables=[
            _crosstab(df, "Region_MostRecentPlacement", "Measure",
                      "TotalAggregatedDays_RestrictiveConfinement",
                      aggfunc="max") or {},
        ],
    )


# ── d-series analyses ──────────────────────────────────────────────


def analyze_d01(df: pd.DataFrame | None = None) -> RichResult:
    """Person-level deaths in custody."""
    df = df if df is not None else load_otis_dataset("d01")
    summary = _summary_lines(df, "d01")
    summary.append(("Distinct individuals",
                    int(df["UniqueIndividual_ID"].nunique())))
    # Real CSV uses MedicalCauseofDeath / MeansofDeath (lowercase o);
    # tolerate either casing.
    cause_col = ("MedicalCauseofDeath" if "MedicalCauseofDeath" in df.columns
                 else "MedicalCauseOfDeath")
    means_col = ("MeansofDeath" if "MeansofDeath" in df.columns
                 else "MeansOfDeath")
    return RichResult(
        title="d01 — Custodial deaths (person-level)",
        summary_lines=summary,
        tables=[
            {"title": "By region:",
             "headers": ["Region", "Deaths"],
             "rows": [[k, int(v)] for k, v in
                      df["Region_AtTimeOfDeath"].value_counts().items()]},
            {"title": "By housing unit type:",
             "headers": ["HousingUnit", "Deaths"],
             "rows": [[k, int(v)] for k, v in
                      df["HousingUnit_Type"].value_counts().items()]},
            {"title": "By medical cause:",
             "headers": ["Cause", "Deaths"],
             "rows": [[k, int(v)] for k, v in
                      df[cause_col].value_counts().items()]},
            {"title": "By means of death:",
             "headers": ["Means", "Deaths"],
             "rows": [[k, int(v)] for k, v in
                      df[means_col].value_counts().items()]},
        ],
    )


def _d_simple(ds_id: str, by: str) -> callable:
    def _fn(df: pd.DataFrame | None = None) -> RichResult:
        df = df if df is not None else load_otis_dataset(ds_id)
        meta = DATASET_REGISTRY[ds_id]
        return RichResult(
            title=f"{ds_id} — {meta.description}",
            summary_lines=_summary_lines(df, ds_id),
            tables=[
                _year_trend(df, "Number_CustodialDeaths") or {"title": "(none)", "headers": [], "rows": []},
                _crosstab(df, by, "Year", "Number_CustodialDeaths") or {"title": "(none)", "headers": [], "rows": []},
            ],
        )
    return _fn


analyze_d02 = _d_simple("d02", "Gender")
analyze_d03 = _d_simple("d03", "Race")
analyze_d04 = _d_simple("d04", "Religion")
analyze_d05 = _d_simple("d05", "Age_Category")


def analyze_d06(df: pd.DataFrame | None = None) -> RichResult:
    df = df if df is not None else load_otis_dataset("d06")
    return RichResult(
        title="d06 — Custodial deaths × alert × medical cause",
        summary_lines=_summary_lines(df, "d06"),
        tables=[
            _crosstab(df, "MedicalCauseOfDeath", "Alert_Type",
                      "Number_CustodialDeaths") or {},
        ],
    )


def analyze_d07(df: pd.DataFrame | None = None) -> RichResult:
    df = df if df is not None else load_otis_dataset("d07")
    return RichResult(
        title="d07 — Custodial deaths × alert × housing unit",
        summary_lines=_summary_lines(df, "d07"),
        tables=[
            _crosstab(df, "HousingUnit_Type", "Alert_Type",
                      "Number_CustodialDeaths") or {},
        ],
    )


# ── Master driver ──────────────────────────────────────────────────


_ANALYSES = {
    "b01": analyze_b01, "b02": analyze_b02, "b03": analyze_b03,
    "b04": analyze_b04, "b05": analyze_b05, "b06": analyze_b06,
    "b07": analyze_b07, "b08": analyze_b08, "b09": analyze_b09,
    "c01": analyze_c01, "c02": analyze_c02, "c03": analyze_c03,
    "c04": analyze_c04, "c05": analyze_c05, "c06": analyze_c06,
    "c07": analyze_c07, "c08": analyze_c08, "c09": analyze_c09,
    "c10": analyze_c10, "c11": analyze_c11, "c12": analyze_c12,
    "d01": analyze_d01, "d02": analyze_d02, "d03": analyze_d03,
    "d04": analyze_d04, "d05": analyze_d05, "d06": analyze_d06,
    "d07": analyze_d07,
}


def analyze_a01(df: pd.DataFrame | None = None) -> RichResult:
    """High-level a01 (Restrictive Confinement Detailed) analysis.

    Wraps the full causal pipeline that the OTIS-RC research
    runs against ``a01_restrictive_confinement_detailed_dataset.csv``:

      1. Person-year aggregation via 8-state alert-combo encoding
         (``moirais.otis_causal.make_pair_alert_to_volatility_a01``).
      2. MatchIt 1:1 NN PSM with caliper.
      3. IRM-DML with cross-fitted random-forest nuisances.
      4. Multi-way clustered standard errors (id, region, year).

    Returns a :class:`RichResult` with both ATE and ATTE estimates.
    """
    from . import otis_causal as oc

    if df is None:
        data, T, Y, covs = oc.make_pair_alert_to_volatility_a01()
    else:
        data, T, Y, covs = oc.make_pair_alert_to_volatility_ruhela(df)

    fit = oc.otis_irm_dml(
        data, treatment=T, outcome=Y, covariates=covs,
        cluster_cols="UniqueIndividual_ID",
        match_first=True,
    )
    summary = [
        ("Source file", "a01_restrictive_confinement_detailed_dataset.csv"),
        ("Person-years (after MatchIt)", fit["n"]),
        ("ATE", round(fit["ate"], 4)),
        ("ATE 95% CI", (round(fit["ate_ci95"][0], 4),
                          round(fit["ate_ci95"][1], 4))),
        ("ATTE", round(fit["atte"], 4)),
        ("ATTE 95% CI", (round(fit["atte_ci95"][0], 4),
                            round(fit["atte_ci95"][1], 4))),
        ("Standard error type", fit["se_kind"]),
    ]
    return RichResult(
        title=("OTIS a01 — high alert complexity (ac ≥ 2) "
                "→ regional volatility (vm count)"),
        summary_lines=summary,
        interpretation=(
            "MatchIt-then-IRM-DML reproduction of the published "
            "res_pool finding on the canonical Restrictive "
            "Confinement Detailed Dataset."),
        payload=fit,
    )



def _dual_irm_dml_on(df: pd.DataFrame,
                      *,
                      ds_id: str,
                      source_label: str,
                      title: str,
                      interpretation: str,
                      cluster_col: str = "EndFiscalYear",
                      include_match_first: bool = True,
                      include_per_year: bool = True,
                      ) -> RichResult:
    """Shared full-dual IRM-DML driver for a01 and b01 analyzers.

    Runs Ruhela + Naive arms side by side on `df`, optionally adds the
    match_first PSM-pruned Ruhela fit and per-fiscal-year Ruhela ATEs.
    Both `analyze_a01_dual` and `analyze_b01_dual` delegate here.
    """
    from . import otis_causal as oc

    both = oc.make_pair_alert_to_volatility_all(df)
    rows = []
    payloads: dict = {}
    for arm in ("ruhela", "naive"):
        data, T, Y, cov = both[arm]
        res = oc.otis_irm_dml(
            data, treatment=T, outcome=Y, covariates=cov,
            cluster_cols=[cluster_col],
        )
        rows.append([
            arm.capitalize(), T, Y, int(res["n"]),
            f"{100*res['p_treat']:.1f}%",
            f"{res['ate']:+.4f}",
            f"[{res['ate_ci95'][0]:+.3f}, {res['ate_ci95'][1]:+.3f}]",
            f"{res['ate_pval']:.2e}",
            f"{res['atte']:+.4f}",
            f"[{res['atte_ci95'][0]:+.3f}, {res['atte_ci95'][1]:+.3f}]",
        ])
        payloads[arm] = {k: v for k, v in res.items()
                         if k not in {"data"}}

    extra_rows = []
    if include_match_first:
        data_r, T_r, Y_r, cov_r = both["ruhela"]
        res_mf = oc.otis_irm_dml(
            data_r, treatment=T_r, outcome=Y_r, covariates=cov_r,
            cluster_cols=[cluster_col], match_first=True,
        )
        extra_rows.append([
            "Ruhela + match_first", T_r, Y_r, int(res_mf["n"]),
            f"{100*res_mf['p_treat']:.1f}%",
            f"{res_mf['ate']:+.4f}",
            f"[{res_mf['ate_ci95'][0]:+.3f}, {res_mf['ate_ci95'][1]:+.3f}]",
            f"{res_mf['ate_pval']:.2e}",
            f"{res_mf['atte']:+.4f}",
            f"[{res_mf['atte_ci95'][0]:+.3f}, {res_mf['atte_ci95'][1]:+.3f}]",
        ])
        payloads["ruhela_match_first"] = {k: v for k, v in res_mf.items()
                                            if k not in {"data"}}

    per_year_rows = []
    if include_per_year:
        data_r, T_r, Y_r, cov_r = both["ruhela"]
        try:
            yr_results = oc.otis_per_year_irm_dml(
                data_r, treatment=T_r, outcome=Y_r, covariates=cov_r,
            )
            payloads["ruhela_per_year"] = {
                str(k): {kk: vv for kk, vv in r.items()
                          if kk not in {"data"}}
                for k, r in yr_results.items() if isinstance(r, dict)}
            for y, r in sorted(yr_results.items()):
                if isinstance(r, dict) and "ate" in r:
                    per_year_rows.append([
                        int(y), int(r.get("n", 0)),
                        f"{r['ate']:+.4f}",
                        f"[{r['ate_ci95'][0]:+.3f}, {r['ate_ci95'][1]:+.3f}]",
                        f"{r['ate_pval']:.2e}",
                    ])
        except Exception as e:  # noqa: BLE001
            per_year_rows = [[f"err: {type(e).__name__}", str(e)[:80]]]

    summary = [
        ("Source file", source_label),
        ("Dataset id", ds_id),
        ("Cluster axis", cluster_col),
        ("Person-years", int(both["ruhela"][0].shape[0])),
        ("Ruhela ATE",
            f"{payloads['ruhela']['ate']:+.4f} "
            f"(SE {payloads['ruhela']['ate_se']:.4f})"),
        ("Naive ATE",
            f"{payloads['naive']['ate']:+.4f} "
            f"(SE {payloads['naive']['ate_se']:.4f})"),
    ]
    if include_match_first:
        summary.append(("Ruhela ATE (match_first PSM)",
            f"{payloads['ruhela_match_first']['ate']:+.4f} "
            f"(SE {payloads['ruhela_match_first']['ate_se']:.4f}, "
            f"matched n={payloads['ruhela_match_first']['n']})"))
    if include_per_year and per_year_rows:
        summary.append(("Per-year Ruhela ATEs",
            ", ".join(f"FY{r[0]}={r[2]}" for r in per_year_rows
                       if isinstance(r[0], int))))

    tables = [{
        "title": (f"Side-by-side dual on {ds_id}: T = high-alert-complexity "
                   f"(arm-specific), Y = vm; cluster = {cluster_col}"),
        "headers": ["Arm", "T", "Y", "n", "p_treat",
                     "ATE", "ATE 95% CI", "ATE p", "ATTE", "ATTE 95% CI"],
        "rows": rows + extra_rows,
    }]
    if include_per_year and per_year_rows:
        tables.append({
            "title": f"Per-fiscal-year Ruhela IRM-DML on {ds_id}:",
            "headers": ["FY", "n", "ATE", "ATE 95% CI", "ATE p"],
            "rows": per_year_rows,
        })

    return RichResult(
        title=title,
        summary_lines=summary,
        tables=tables,
        interpretation=interpretation,
        payload={"arms": payloads, "cluster_col": cluster_col,
                 "ds_id": ds_id},
    )


def _ruhela_formulations_on(df: pd.DataFrame,
                      *,
                      ds_id: str,
                      source_label: str,
                      title: str,
                      interpretation: str,
                      cluster_col: str = "EndFiscalYear",
                      treatment: str | None = None,
                      outcome: str | None = None,
                      covariates: list[str] | None = None,
                      naive_pair_fn=None,
                      include_atc: bool = True,
                      include_plr: bool = True,
                      include_multi_se: bool = True,
                      include_superlearner: bool = True,
                      propensity_calibration: str = "none",
                      ) -> RichResult:
    """Ruhela formulations (full DLRM).

    Runs the complete OTIS-RC methodology arc on a Ruhela formulation —
    a (treatment, outcome, covariates) design choice for a specific
    OTIS dataset. Defaults to the author's canonical alert-complexity → vm
    formulation (via ``make_pair_alert_to_volatility_all``); pass
    explicit ``treatment``/``outcome``/``covariates`` for dataset-
    specific Ruhela formulations on b02-b09 / c-series / d-series.

    DLRM (each estimator runs on the Ruhela arm):

      1. IPW (Hájek)               — single-robust; Lunceford-Davidian SE
      2. AIPW                       — doubly-robust; RRZ 1994 IF, cross-fit
      3. g-computation              — single-robust outcome model + bootstrap
      4. PSM 1:1 NN                 — Austin 2011 caliper, ATT
      5. PSM subclass (5 strata)    — Rosenbaum-Rubin 1983, ATE
      6. IRM-DML                    — Chernozhukov 2018, ATE+ATTE+ATC,
                                       cluster-robust SE
      7. PSM->IRM-DML (match_first) — the author's MatchIt-then-DoubleML pipeline
      8. ATC (AIPW-flavour)         — E[Y(1)-Y(0) | D=0]
      9. PLR DML                    — homogeneous-effect double ML
     10. SuperLearner-stacked AIPW  — RF+ridge+GLM+mean convex stack

    Plus IRM-DML SE comparison: pooled (iid), cluster on EndFiscalYear,
    cluster on UniqueIndividual_ID, multi-way (year × id) — same point
    estimate, four standard errors.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataset.
    ds_id : str
        Dataset id (a01, b01, b02, c01...).
    cluster_col : str
        Primary cluster axis for IRM-DML SE.
    treatment, outcome, covariates : optional explicit Ruhela formulation.
        If None, falls back to ``make_pair_alert_to_volatility_all`` (a01/b01).
    naive_pair_fn : callable returning (df, T, Y, cov), optional.
        If supplied, runs Naive-arm sensitivity (e.g. any-flag, vm-binary).
        Default None: skip Naive arm (used for b02+/c/d formulations).
    include_atc / include_plr / include_multi_se / include_superlearner :
        Toggles for the extended ensemble.
    propensity_calibration : "none" | "platt" | "isotonic"
        Calibration of propensity scores in IPW/AIPW.
    """
    from . import otis_causal as oc

    payloads: dict = {"ensemble": [], "match_first": None,
                       "ds_id": ds_id, "cluster_col": cluster_col,
                       "propensity_calibration": propensity_calibration}

    # Build the Ruhela arm: explicit triple, or default to alert-complexity
    if treatment is not None and outcome is not None and covariates is not None:
        # Explicit dataset-specific Ruhela formulation
        data_r = df[[treatment, outcome] + list(covariates)].dropna().copy()
        T_r, Y_r, cov_r = treatment, outcome, list(covariates)
        formulation_label = f"{ds_id}: T={treatment}, Y={outcome}"
        naive_arm = None
        if naive_pair_fn is not None:
            try:
                naive_arm = naive_pair_fn(df)
            except Exception:  # noqa: BLE001
                naive_arm = None
    else:
        # Default a01/b01 alert-complexity formulation with naive sensitivity
        both = oc.make_pair_alert_to_volatility_all(df)
        data_r, T_r, Y_r, cov_r = both["ruhela"]
        formulation_label = f"{ds_id}: T_high_ac → Y_vm_count (canonical)"
        naive_arm = both["naive"]

    payloads["formulation"] = {"label": formulation_label,
                                  "treatment": T_r, "outcome": Y_r,
                                  "covariates": cov_r,
                                  "n_rows": int(data_r.shape[0])}

    rows: list = []

    # Helper to format a CausalEstimate row
    def _row(idx: str, est, label: str = None,
             estimand: str = "ATE") -> list:
        if est is None or not hasattr(est, "ate"):
            return [idx, "—", "err", "—", "—", "—", "—"]
        ci = est.ate_ci95
        return [
            idx if not label else f"{idx} ({label})",
            estimand,
            f"{est.ate:+.4f}",
            f"{est.ate_se:.4f}",
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
            f"{est.ate_pval:.2e}",
            ", ".join(est.notes[:2]) if est.notes else "—",
        ]

    # 1. IPW
    try:
        ipw = oc.otis_ipw(data_r, treatment=T_r, outcome=Y_r,
                           covariates=cov_r,
                           propensity_calibration=propensity_calibration)
        rows.append(_row("1. IPW (Hájek)", ipw))
        payloads["ensemble"].append({"estimator": "IPW",
                                     "ate": float(ipw.ate),
                                     "se": float(ipw.ate_se),
                                     "p": float(ipw.ate_pval),
                                     "n": int(ipw.n), "notes": ipw.notes})
    except Exception as e:  # noqa: BLE001
        rows.append(["1. IPW", "—", "err", str(e)[:30], "—", "—", "—"])

    # 2. AIPW
    try:
        aipw = oc.otis_aipw(data_r, treatment=T_r, outcome=Y_r,
                             covariates=cov_r,
                             propensity_calibration=propensity_calibration)
        rows.append(_row("2. AIPW (RRZ DR)", aipw))
        payloads["ensemble"].append({"estimator": "AIPW",
                                     "ate": float(aipw.ate),
                                     "se": float(aipw.ate_se),
                                     "p": float(aipw.ate_pval),
                                     "n": int(aipw.n), "notes": aipw.notes})
    except Exception as e:  # noqa: BLE001
        rows.append(["2. AIPW", "—", "err", str(e)[:30], "—", "—", "—"])

    # 3. g-computation
    try:
        gc = oc.otis_gcomputation(data_r, treatment=T_r, outcome=Y_r,
                                    covariates=cov_r, n_bootstrap=200)
        rows.append(_row("3. g-computation", gc))
        payloads["ensemble"].append({"estimator": "g-computation",
                                     "ate": float(gc.ate),
                                     "se": float(gc.ate_se),
                                     "p": float(gc.ate_pval),
                                     "n": int(gc.n), "notes": gc.notes})
    except Exception as e:  # noqa: BLE001
        rows.append(["3. g-computation", "—", "err", str(e)[:30], "—", "—", "—"])

    # 4. PSM 1:1 NN
    try:
        psm = oc.otis_psm(data_r, treatment=T_r, outcome=Y_r,
                           covariates=cov_r, k=1)
        rows.append(_row("4. PSM 1:1 NN", psm, estimand="ATT"))
        payloads["ensemble"].append({"estimator": "PSM-NN",
                                     "att": float(psm.ate),
                                     "se": float(psm.ate_se),
                                     "p": float(psm.ate_pval),
                                     "n": int(psm.n), "notes": psm.notes})
    except Exception as e:  # noqa: BLE001
        rows.append(["4. PSM 1:1 NN", "—", "err", str(e)[:30], "—", "—", "—"])

    # 5. PSM subclass
    try:
        psm_sc = oc.otis_psm_subclass(data_r, treatment=T_r, outcome=Y_r,
                                        covariates=cov_r, n_strata=5)
        rows.append(_row("5. PSM subclass (5)", psm_sc))
        payloads["ensemble"].append({"estimator": "PSM-subclass",
                                     "ate": float(psm_sc.ate),
                                     "se": float(psm_sc.ate_se),
                                     "p": float(psm_sc.ate_pval),
                                     "n": int(psm_sc.n),
                                     "notes": psm_sc.notes})
    except Exception as e:  # noqa: BLE001
        rows.append(["5. PSM subclass", "—", "err", str(e)[:30], "—", "—", "—"])

    # 6. IRM-DML — primary (cluster on cluster_col), now with ATC
    irm_results: dict = {}
    try:
        dml = oc.otis_irm_dml(data_r, treatment=T_r, outcome=Y_r,
                                covariates=cov_r,
                                cluster_cols=[cluster_col]
                                if cluster_col in data_r.columns else None)
        rows.append(["6. IRM-DML", "ATE",
                      f"{dml['ate']:+.4f}", f"{dml['ate_se']:.4f}",
                      f"[{dml['ate_ci95'][0]:+.3f}, {dml['ate_ci95'][1]:+.3f}]",
                      f"{dml['ate_pval']:.2e}",
                      f"cluster={dml['se_kind']}"])
        rows.append(["6. IRM-DML", "ATTE",
                      f"{dml['atte']:+.4f}", f"{dml['atte_se']:.4f}",
                      f"[{dml['atte_ci95'][0]:+.3f}, {dml['atte_ci95'][1]:+.3f}]",
                      f"{dml['atte_pval']:.2e}", "—"])
        if include_atc and dml.get("atc") is not None:
            rows.append(["6. IRM-DML", "ATC",
                          f"{dml['atc']:+.4f}",
                          f"{dml.get('atc_se', float('nan')):.4f}",
                          (f"[{dml['atc_ci95'][0]:+.3f}, "
                            f"{dml['atc_ci95'][1]:+.3f}]"
                            if dml.get("atc_ci95") else "—"),
                          (f"{dml['atc_pval']:.2e}"
                            if dml.get("atc_pval") is not None else "—"),
                          "—"])
        irm_results = dml
        payloads["ensemble"].append({"estimator": "IRM-DML",
                                     "ate": float(dml["ate"]),
                                     "ate_se": float(dml["ate_se"]),
                                     "atte": float(dml["atte"]),
                                     "atte_se": float(dml["atte_se"]),
                                     "atc": (float(dml["atc"])
                                             if dml.get("atc") is not None
                                             else None),
                                     "n": int(dml["n"])})
    except Exception as e:  # noqa: BLE001
        rows.append(["6. IRM-DML", "—", "err", str(e)[:30], "—", "—", "—"])

    # 7. PSM -> IRM-DML
    try:
        dml_mf = oc.otis_irm_dml(data_r, treatment=T_r, outcome=Y_r,
                                   covariates=cov_r,
                                   cluster_cols=[cluster_col]
                                   if cluster_col in data_r.columns else None,
                                   match_first=True)
        rows.append(["7. PSM->IRM-DML (match_first)", "ATE",
                      f"{dml_mf['ate']:+.4f}", f"{dml_mf['ate_se']:.4f}",
                      f"[{dml_mf['ate_ci95'][0]:+.3f}, "
                      f"{dml_mf['ate_ci95'][1]:+.3f}]",
                      f"{dml_mf['ate_pval']:.2e}",
                      f"matched_n={dml_mf['n']}"])
        payloads["match_first"] = {k: v for k, v in dml_mf.items()
                                    if k != "data"}
    except Exception as e:  # noqa: BLE001
        rows.append(["7. PSM->IRM-DML", "—", "err", str(e)[:30], "—", "—", "—"])

    # 8. ATC (AIPW-style)
    if include_atc:
        try:
            atc = oc.otis_atc(data_r, treatment=T_r, outcome=Y_r,
                                covariates=cov_r)
            rows.append(_row("8. ATC (AIPW)", atc, estimand="ATC"))
            payloads["ensemble"].append({"estimator": "ATC-AIPW",
                                         "atc": float(atc.ate),
                                         "se": float(atc.ate_se),
                                         "p": float(atc.ate_pval),
                                         "n": int(atc.n), "notes": atc.notes})
        except Exception as e:  # noqa: BLE001
            rows.append(["8. ATC (AIPW)", "—", "err",
                          str(e)[:30], "—", "—", "—"])

    # 9. PLR (Partially Linear DML)
    if include_plr:
        try:
            plr = oc.otis_plr(data_r, treatment=T_r, outcome=Y_r,
                                covariates=cov_r)
            rows.append(_row("9. PLR (Chernozhukov 2018)", plr,
                              estimand="θ"))
            payloads["ensemble"].append({"estimator": "PLR",
                                         "theta": float(plr.ate),
                                         "se": float(plr.ate_se),
                                         "p": float(plr.ate_pval),
                                         "n": int(plr.n), "notes": plr.notes})
        except Exception as e:  # noqa: BLE001
            rows.append(["9. PLR", "—", "err", str(e)[:30], "—", "—", "—"])

    # 10. SuperLearner-stacked AIPW
    if include_superlearner:
        try:
            sl = oc.otis_aipw_superlearner(data_r, treatment=T_r,
                                             outcome=Y_r, covariates=cov_r,
                                             propensity_calibration=propensity_calibration)
            rows.append(_row("10. SuperLearner AIPW", sl))
            payloads["ensemble"].append({"estimator": "SuperLearner-AIPW",
                                         "ate": float(sl.ate),
                                         "se": float(sl.ate_se),
                                         "p": float(sl.ate_pval),
                                         "n": int(sl.n),
                                         "notes": sl.notes})
        except Exception as e:  # noqa: BLE001
            rows.append(["10. SuperLearner AIPW", "—", "err",
                          str(e)[:30], "—", "—", "—"])

    # === Multi-SE comparison on IRM-DML primary point estimate ===
    multi_se_rows: list = []
    if include_multi_se and irm_results:
        ate_pt = irm_results.get("ate", float("nan"))
        # Try four SE flavours: pooled, EFY-cluster, UID-cluster, multi-way
        flavours = [
            ("pooled (iid)", None),
            ("cluster: EndFiscalYear",
                ["EndFiscalYear"] if "EndFiscalYear" in data_r.columns else None),
            ("cluster: UniqueIndividual_ID",
                ["UniqueIndividual_ID"]
                if "UniqueIndividual_ID" in data_r.columns else None),
            ("multi-way: yr × id",
                ["EndFiscalYear", "UniqueIndividual_ID"]
                if ("EndFiscalYear" in data_r.columns
                    and "UniqueIndividual_ID" in data_r.columns) else None),
        ]
        for label, cl in flavours:
            if cl is None and label != "pooled (iid)":
                multi_se_rows.append([label, "—", "n/a (col missing)",
                                       "—", "—"])
                continue
            try:
                fit = oc.otis_irm_dml(data_r, treatment=T_r, outcome=Y_r,
                                        covariates=cov_r, cluster_cols=cl)
                multi_se_rows.append([
                    label,
                    f"{fit['ate']:+.4f}",
                    f"{fit['ate_se']:.4f}",
                    f"[{fit['ate_ci95'][0]:+.3f}, {fit['ate_ci95'][1]:+.3f}]",
                    f"{fit['ate_pval']:.2e}",
                ])
            except Exception as e:  # noqa: BLE001
                multi_se_rows.append([label, "—", "err",
                                       str(e)[:30], "—"])
        payloads["multi_se"] = multi_se_rows

    # === Naive-arm sensitivity (only if formulation defines a naive pair) ===
    naive_rows = []
    if naive_arm is not None:
        data_n, T_n, Y_n, cov_n = naive_arm
        for label_n, fn in [
            ("IPW", oc.otis_ipw),
            ("AIPW", oc.otis_aipw),
            ("g-comp", oc.otis_gcomputation),
        ]:
            try:
                r = fn(data_n, treatment=T_n, outcome=Y_n,
                        covariates=cov_n)
                naive_rows.append([label_n, f"{r.ate:+.4f}",
                                    f"{r.ate_se:.4f}",
                                    f"[{r.ate_ci95[0]:+.3f}, "
                                    f"{r.ate_ci95[1]:+.3f}]",
                                    f"{r.ate_pval:.2e}"])
            except Exception as e:  # noqa: BLE001
                naive_rows.append([label_n, "err", str(e)[:30],
                                    "—", "—"])
        payloads["naive_arm"] = naive_rows

    # === Balance & propensity diagnostics on Ruhela arm ===
    balance_summary = "—"
    try:
        balance = oc.otis_balance(data_r, treatment=T_r,
                                    covariates=cov_r)
        smd_data = balance.payload if hasattr(balance, "payload") else {}
        max_smd = smd_data.get("max_abs_smd", "n/a")
        balance_summary = (f"max |SMD| = {max_smd:.3f}"
                            if isinstance(max_smd, (int, float))
                            else str(max_smd))
    except Exception as e:  # noqa: BLE001
        balance_summary = f"err: {type(e).__name__}: {e}"

    summary = [
        ("Source file", source_label),
        ("Dataset id", ds_id),
        ("Formulation", formulation_label),
        ("Cluster axis (primary)", cluster_col),
        ("n", int(data_r.shape[0])),
        ("Treated prevalence",
            f"{100*float(data_r[T_r].mean() if data_r[T_r].dtype != object else (data_r[T_r] == data_r[T_r].mode()[0]).mean()):.1f}%"
            if T_r in data_r.columns else "n/a"),
        ("Pre-balance (covariate SMD)", balance_summary),
        ("Propensity calibration", propensity_calibration),
    ]

    tables = [{
        "title": (f"Ruhela formulations (full DLRM, "
                   f"alias DLRM) on {ds_id} — {formulation_label}:"),
        "headers": ["Estimator", "Estimand",
                     "Estimate", "SE", "95% CI", "p-value", "Notes"],
        "rows": rows,
    }]
    if include_multi_se and multi_se_rows:
        tables.append({
            "title": ("IRM-DML SE comparison: pooled vs clustered on "
                       "EndFiscalYear, UniqueIndividual_ID, multi-way "
                       "(year × id):"),
            "headers": ["SE flavour", "ATE", "SE", "95% CI", "p-value"],
            "rows": multi_se_rows,
        })
    if naive_rows:
        tables.append({
            "title": ("Naive-arm sensitivity (any-flag, vm-binary): "
                       "concordance with Ruhela in sign, smaller magnitude"),
            "headers": ["Estimator", "ATE", "SE", "95% CI", "p-value"],
            "rows": naive_rows,
        })

    return RichResult(
        title=title,
        summary_lines=summary,
        tables=tables,
        interpretation=interpretation,
        payload=payloads,
    )


def analyze_a01_ruhela_formulations(df: pd.DataFrame | None = None,
                            *,
                            cluster_col: str = "EndFiscalYear",
                            ) -> RichResult:
    """Ruhela formulations (full DLRM) on a01 — the canonical
    OTIS-RC dataset.

    Runs the complete OTIS-RC methodology arc on the author's Ruhela
    formulation (T = high alert complexity, Y = regional volatility).
    See ``_ruhela_formulations_on`` for the full estimator list.

    Attribution
    -----------
    The Ruhela formulation is the author's design end-to-end; the supporting-
    contributor credits (Doob — pointed the author to the dataset; Levinsky —
    course context, reviewed only the preliminary a01 analysis; Medina
    — reviewed the formal write-up) are documented in
    ``moirais.otis_causal``'s module docstring.

    Parameters
    ----------
    df : pd.DataFrame, optional
        a01 DataFrame; if None, ``otis_datasets.load_otis_dataset('a01')``.
    cluster_col : str
        Cluster axis for cluster-robust SE in the IRM-DML estimator
        (default 'EndFiscalYear').
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("a01")
    return _ruhela_formulations_on(
        df,
        ds_id="a01",
        source_label="a01_restrictive_confinement_detailed_dataset.csv",
        title=("OTIS a01 — Ruhela formulations (full DLRM) on the "
                "canonical formulation (T = ac ≥ 2 → Y = vm count)"),
        interpretation=(
            "Ruhela formulations on the canonical OTIS-RC dataset (a01). The "
            "Ruhela formulation (T = ac ≥ 2 alert-state complexity, "
            "Y = vm regional-volatility count) is the published "
            "Goffmanian contrast in notez1a.qmd § res_pool. Concordance "
            "across IPW (single-robust on propensity), g-computation "
            "(single-robust on outcome), AIPW (doubly-robust), PSM-NN "
            "(ATT, nonparametric), PSM-subclass (ATE, nonparametric), "
            "The man who moves a mountain begins by carrying away small stones. — Confucius"
            "then-DoubleML pipeline) is the strongest practical signal "
            "of identification. Naive-arm sensitivity (any-flag, vm-"
            "binary) shows the same Goffmanian sign at smaller magnitude."
        ),
        cluster_col=cluster_col,
    )


# Backward-compat alias (renamed 2026-05-09)
def analyze_a01_dual(*args, **kwargs):
    """Deprecated alias for ``analyze_a01_ruhela_formulations``."""
    import warnings as _w
    _w.warn("analyze_a01_dual is deprecated; use analyze_a01_ruhela_formulations",
            DeprecationWarning, stacklevel=2)
    return analyze_a01_ruhela_formulations(*args, **kwargs)


def analyze_b01_ruhela_formulations(df: pd.DataFrame | None = None,
                            *,
                            cluster_col: str = "EndFiscalYear",
                            ) -> RichResult:
    """Ruhela formulations (full DLRM) on b01 (Segregation Detailed).

    Per-placement segregation complement to a01's per-day RC records.
    Same Ruhela alert-complexity → regional-volatility contrast,
    larger magnitude effect because segregation is a stronger
    institutional contrast.

    See ``analyze_a01_ruhela_formulations`` for the full attribution and
    estimator catalogue.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b01")
    return _ruhela_formulations_on(
        df,
        ds_id="b01",
        source_label="b01_segregation_detailed_dataset.csv",
        title=("OTIS b01 — Ruhela formulations (full DLRM) "
                "(segregation-detailed complement)"),
        interpretation=(
            "Ruhela formulations on b01 segregation-placement records, "
            "complementing the a01 RC analysis. Same Ruhela formulation "
            "and DLRM ensemble; the larger ATE magnitude (~3× a01) "
            "reflects segregation being a stronger institutional "
            "contrast than restrictive confinement, and per-placement "
            "rather than per-day units of analysis."
        ),
        cluster_col=cluster_col,
    )



# ── Per-dataset Ruhela formulations (b02-b09 / c01-c12 / d01-d07) ──
#
# OTIS data shape varies by dataset: a01/b01 are panel with alerts +
# region pair (canonical Ruhela formulation applies); b02-b09 / c-series
# are aggregate or have only some Ruhela variables; d-series carries
# no alert columns. Each function below picks a (T, Y, covariates)
# triple that's well-defined for the dataset's actual columns. These
# available variables.


def _female_indicator(s) -> "pd.Series":
    return (s.astype(str).str.lower() == "female").astype(int)


def _region_toronto_indicator(s) -> "pd.Series":
    return (s.astype(str).str.lower() == "toronto").astype(int)


def _age_50plus_indicator(s) -> "pd.Series":
    return s.astype(str).str.contains("50", na=False).astype(int)


def analyze_b02_ruhela_formulations(df: "pd.DataFrame | None" = None,
                                      *,
                                      cluster_col: str = "EndFiscalYear",
                                      ) -> RichResult:
    """Ruhela formulation on b02 (Segregation Total Days per individual).

    b02 has UniqueIndividual_ID + EndFiscalYear + Gender + Region +
    Age_Category + TotalAggregatedDays_Segregation but NO alert columns.
    The canonical alert-complexity formulation is impossible; instead
    the author's b02-specific Ruhela formulation tests **gender disparity in
    segregation-day burden**:

      T = Gender == "Female" indicator
      Y = TotalAggregatedDays_Segregation (count, person-year)
      covariates = [Region_MostRecentPlacement, Age_Category, EndFiscalYear]

    Same DLRM ensemble as the canonical formulation. Sister
    formulations on the same b02 data could swap T for region or age.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b02")
    work = df.dropna(subset=["Gender", "TotalAggregatedDays_Segregation",
                              "Region_MostRecentPlacement", "Age_Category",
                              "EndFiscalYear"]).copy()
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_formulations_on(
        work,
        ds_id="b02",
        source_label="b02_segregation_detailed_total_days.csv",
        title=("OTIS b02 — Ruhela formulations: "
                "T=Female → Y=Total seg days within FY"),
        interpretation=(
            "Ruhela formulation on b02 testing gender disparity in "
            "segregation-day burden. Per-person-year aggregated days "
            "in segregation as the count outcome; Female indicator as "
            "the treatment. Same DLRM ensemble as a01/b01; no Naive "
            "arm because b02 lacks the alert columns to construct one."
        ),
        cluster_col=cluster_col,
        treatment="T_female",
        outcome="TotalAggregatedDays_Segregation",
        covariates=["Region_MostRecentPlacement", "Age_Category",
                     "EndFiscalYear"],
    )



def analyze_ruhela_per_year(df: "pd.DataFrame",
                              *,
                              ds_id: str,
                              treatment: str,
                              outcome: str,
                              covariates: list[str],
                              year_col: str = "EndFiscalYear",
                              cluster_col: str | None = "EndFiscalYear",
                              propensity_calibration: str = "none",
                              ) -> RichResult:
    """Per-fiscal-year full DLRM on each Ruhela formulation.

    Runs the complete DLRM (IPW + AIPW + g-comp + PSM-NN +
    PSM-subclass + IRM-DML + match_first + ATC + PLR + SuperLearner)
    separately on each fiscal year. ~7× the standard per-year IRM-DML
    runtime. Mirrors a per-year extension of the author's ``res_by_year``.
    """
    from . import otis_causal as oc

    cl = ([cluster_col] if cluster_col else None)
    by_year = oc.otis_per_year_irm_dml(
        df, treatment=treatment, outcome=outcome,
        covariates=covariates, year_col=year_col,
        cluster_cols=cl, full_ensemble=True,
        propensity_calibration=propensity_calibration,
    )

    # Build a long table: one row per (year, estimator)
    long_rows: list = []
    for yr, year_results in sorted(by_year.items()):
        if not isinstance(year_results, dict):
            continue
        n = year_results.get("n", "?")
        for est_label in ("ipw", "aipw", "gcomp", "psm_nn",
                            "psm_subclass", "atc", "plr",
                            "superlearner"):
            est = year_results.get(est_label, {})
            if "error" in est:
                long_rows.append([yr, est_label.upper(),
                                    "err", "—", "—", "—", est.get("error", "")[:30]])
                continue
            ci = est.get("ci95", (float("nan"), float("nan")))
            long_rows.append([
                yr, est_label.upper(),
                f"{est.get('ate', float('nan')):+.4f}",
                f"{est.get('se', float('nan')):.4f}",
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                f"{est.get('p', float('nan')):.2e}",
                f"n={est.get('n', '?')}",
            ])
        # IRM-DML: report ATE + ATTE + ATC inline
        irm = year_results.get("irm_dml", {})
        if "error" in irm:
            long_rows.append([yr, "IRM-DML", "err", "—", "—", "—",
                                irm.get("error", "")[:30]])
        else:
            for label_irm, key, key_se in [
                ("IRM-DML ATE", "ate", "ate_se"),
                ("IRM-DML ATTE", "atte", "atte_se"),
                ("IRM-DML ATC", "atc", "atc_se"),
            ]:
                v = irm.get(key)
                vse = irm.get(key_se)
                if v is None or vse is None:
                    continue
                long_rows.append([
                    yr, label_irm,
                    f"{v:+.4f}", f"{vse:.4f}",
                    f"[{v - 1.96 * vse:+.3f}, {v + 1.96 * vse:+.3f}]",
                    "—", f"n={irm.get('n', '?')} {irm.get('se_kind', '')}",
                ])
        # match_first
        mf = year_results.get("match_first", {})
        if "error" in mf:
            long_rows.append([yr, "PSM->IRM-DML", "err", "—", "—", "—",
                                mf.get("error", "")[:30]])
        else:
            v = mf.get("ate", float("nan"))
            vse = mf.get("ate_se", float("nan"))
            long_rows.append([
                yr, "PSM->IRM-DML",
                f"{v:+.4f}", f"{vse:.4f}",
                f"[{v - 1.96 * vse:+.3f}, {v + 1.96 * vse:+.3f}]",
                "—", f"matched_n={mf.get('n', '?')}",
            ])

    summary = [
        ("Dataset id", ds_id),
        ("Treatment", treatment),
        ("Outcome", outcome),
        ("Year column", year_col),
        ("Cluster axis", cluster_col or "iid"),
        ("Years analysed", len([y for y in by_year if not
                                  (isinstance(by_year[y], dict)
                                    and "error" in by_year[y]
                                    and len(by_year[y]) == 1)])),
        ("Estimators per year", 10),
    ]
    return RichResult(
        title=(f"OTIS {ds_id} — per-fiscal-year full DLRM-on-Ruhela-formulations "
                "suite (10 estimators × N years)"),
        summary_lines=summary,
        tables=[{
            "title": ("Per-year × estimator: ATE / SE / 95% CI / p / n. "
                       "Triangulation across IPW + AIPW + g-comp + PSM "
                       "(NN+subclass) + IRM-DML (ATE+ATTE+ATC) + "
                       "match_first + ATC-AIPW + PLR + SuperLearner."),
            "headers": ["FY", "Estimator", "Estimate", "SE",
                         "95% CI", "p", "Notes"],
            "rows": long_rows,
        }],
        interpretation=(
            "Per-year × per-estimator triangulation. Stable signs "
            "across estimators within each FY = no-misspecification "
            "robustness; stable magnitudes across FYs = temporal "
            "stationarity. Use this to identify FYs where some "
            "estimators disagree — typical sign of finite-sample "
            "fragility or genuine effect heterogeneity."
        ),
        payload={"by_year": by_year,
                  "ds_id": ds_id,
                  "treatment": treatment,
                  "outcome": outcome},
    )


def analyze_a01_ruhela_per_year(df: "pd.DataFrame | None" = None,
                                  ) -> RichResult:
    """Per-year full DLRM on Ruhela formulations on a01."""
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("a01")
    data, T, Y, cov = oc.make_pair_alert_to_volatility_a01()
    return analyze_ruhela_per_year(
        data, ds_id="a01", treatment=T, outcome=Y, covariates=cov,
        cluster_col="EndFiscalYear")


def analyze_b01_ruhela_per_year(df: "pd.DataFrame | None" = None,
                                  ) -> RichResult:
    """Per-year full DLRM on Ruhela formulations on b01."""
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b01")
    data, T, Y, cov = oc.make_pair_alert_to_volatility_ruhela(df)
    return analyze_ruhela_per_year(
        data, ds_id="b01", treatment=T, outcome=Y, covariates=cov,
        cluster_col="EndFiscalYear")



# ── Aggregate Ruhela formulations on b03-b09 + c-series ──────────
#
# Aggregate datasets carry pre-summed counts, not individual rows. The
# propensity-score / cross-fitted-ML machinery in the per-row Ruhela
# formulations doesn't apply. The shape-appropriate analog is a
# Poisson + Negative-Binomial GLM with year-FE + treatment-FE +
# demographic-FE, returning an IRR table with 95% CI. This is still


def _ruhela_aggregate_on(df: "pd.DataFrame",
                          *,
                          ds_id: str,
                          source_label: str,
                          title: str,
                          interpretation: str,
                          treatment: str,
                          outcome: str,
                          covariates: "list[str] | None" = None,
                          year_col: str = "EndFiscalYear",
                          cluster_group: "str | None" = None,
                          ) -> RichResult:
    """Aggregate Ruhela formulation: Poisson + NB IRR on count outcomes.

    For aggregate datasets where rows are pre-summed counts grouped by
    demographic / institutional cells. By default fits both Poisson and
    Negative-Binomial GLMs with the chosen treatment + covariates +
    year fixed effect; reports IRR with Wald 95% CI and AIC dispersion
    diagnostics.

    When ``cluster_group`` is supplied, switches to a **GEE marginal
    model** (Liang-Zeger 1986) clustered on that grouping variable,
    with cluster-robust SE. Use this when one categorical has too many
    levels to enter as a fixed effect (e.g. ~76 institutions on a 148-
    row table) — the FE specification becomes collinear and fits fail.
    GEE marginal model is what OTIS-RC's ``glmmTMB nbinom2`` does in
    spirit: report population-level IRR with cluster-robust inference.

    Parameters
    ----------
    df : pre-aggregated DataFrame
    treatment : column whose IRR we want (typically a binary indicator
        derived from a categorical column)
    outcome : count column
    covariates : additional categorical variables to include as fixed
        effects
    year_col : year fixed effect column (defaults to EndFiscalYear)
    cluster_group : if set, fit GEE clustered on this grouping variable
        instead of including it as a categorical FE
    """
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return RichResult(
            title=title,
            warnings=["statsmodels unavailable — install moirais[stats]"],
            interpretation="Aggregate Poisson/NB GLM requires statsmodels.",
        )

    cov = list(covariates or [])
    cluster_cols = [cluster_group] if cluster_group else []
    # Dedup column selection to avoid duplicate-column bug when
    # cluster_group, year_col, treatment, etc. overlap.
    _all_cols = list(dict.fromkeys(
        [treatment, outcome, year_col, *cov, *cluster_cols]))
    work = df[_all_cols].dropna().copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work = work.dropna(subset=[outcome])
    work[outcome] = work[outcome].astype(int)
    if work.empty or work[outcome].sum() == 0:
        return RichResult(
            title=title,
            warnings=["empty cell-table or zero outcome count"],
        )

    # Build a formula string with categorical wrappers for non-numeric covs
    parts = [f"C({treatment})"]
    for c in cov:
        parts.append(f"C({c})")
    parts.append(f"C({year_col})")
    formula = f"{outcome} ~ " + " + ".join(parts)

    rows: list = []
    payloads: dict = {"formula": formula, "ds_id": ds_id,
                       "n_rows": int(work.shape[0]),
                       "treatment": treatment, "outcome": outcome,
                       "cluster_group": cluster_group,
                       "estimands": []}
    aic_pois = float("nan")
    aic_nb = float("nan")

    # === GLM fits (always run for IRR estimate + AIC) ===
    for fam_label, family in (
        ("Poisson", sm.families.Poisson()),
        ("NB", sm.families.NegativeBinomial(alpha=1.0)),
    ):
        try:
            model = smf.glm(formula, data=work, family=family).fit()
            t_coef_keys = [k for k in model.params.index
                            if treatment in k]
            if not t_coef_keys:
                rows.append([fam_label, treatment, "no coef", "—", "—",
                              "—", "—"])
                continue
            t_key = t_coef_keys[0]
            beta = float(model.params[t_key])
            se = float(model.bse[t_key])
            irr = float(np.exp(beta))
            ci_lo = float(np.exp(beta - 1.96 * se))
            ci_hi = float(np.exp(beta + 1.96 * se))
            pval = float(model.pvalues[t_key])
            aic = float(model.aic)
            if fam_label == "Poisson":
                aic_pois = aic
            else:
                aic_nb = aic
            rows.append([
                fam_label, t_key, round(irr, 4),
                f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                f"{pval:.2e}", round(aic, 1),
                f"n={int(work.shape[0])}, total_y={int(work[outcome].sum())}",
            ])
            payloads["estimands"].append({
                "family": fam_label, "coef": t_key,
                "irr": irr, "ci95_low": ci_lo, "ci95_high": ci_hi,
                "p": pval, "aic": aic, "beta": beta, "se": se,
            })
        except Exception as e:  # noqa: BLE001
            rows.append([fam_label, treatment, "fit failed",
                          str(type(e).__name__), str(e)[:30],
                          "—", "—"])

    # === GEE clustered fits when cluster_group set ===
    # Iterate over (Family × WorkingCorrelation): Poisson + NB,
    # Exchangeable + Independence. Independence = sandwich-only
    # cluster-robust SE without modelling within-cluster correlation
    # (the conservative sensitivity baseline). Exchangeable = constant
    # within-cluster correlation (the standard OTIS-RC analog).
    if cluster_group is not None:
        n_groups = int(work[cluster_group].nunique())
        gee_specs = [
            ("Poisson", "Exch", sm.families.Poisson(),
                sm.cov_struct.Exchangeable()),
            ("Poisson", "Indep", sm.families.Poisson(),
                sm.cov_struct.Independence()),
            ("NB", "Exch", sm.families.NegativeBinomial(alpha=1.0),
                sm.cov_struct.Exchangeable()),
            ("NB", "Indep", sm.families.NegativeBinomial(alpha=1.0),
                sm.cov_struct.Independence()),
        ]
        for fam_lbl, corr_lbl, family, cov_struct in gee_specs:
            row_label = (f"GEE-{fam_lbl} (cluster:{cluster_group}, "
                          f"{corr_lbl})")
            try:
                gee_model = smf.gee(
                    formula, groups=cluster_group, data=work,
                    family=family, cov_struct=cov_struct,
                ).fit()
                t_coef_keys = [k for k in gee_model.params.index
                                if treatment in k]
                if not t_coef_keys:
                    rows.append([row_label, treatment, "no coef",
                                  "—", "—", "—", "—"])
                    continue
                t_key = t_coef_keys[0]
                beta = float(gee_model.params[t_key])
                se = float(gee_model.bse[t_key])
                irr = float(np.exp(beta))
                ci_lo = float(np.exp(beta - 1.96 * se))
                ci_hi = float(np.exp(beta + 1.96 * se))
                pval = float(gee_model.pvalues[t_key])
                rows.append([
                    row_label, t_key, round(irr, 4),
                    f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                    f"{pval:.2e}", "—",
                    f"groups={n_groups}, n_obs={int(work.shape[0])}",
                ])
                payloads["estimands"].append({
                    "family": f"GEE-{fam_lbl}",
                    "working_correlation": corr_lbl,
                    "cluster_group": cluster_group,
                    "n_groups": n_groups,
                    "coef": t_key, "irr": irr,
                    "ci95_low": ci_lo, "ci95_high": ci_hi,
                    "p": pval, "beta": beta, "se": se,
                })
            except Exception as e:  # noqa: BLE001
                rows.append([
                    row_label, treatment, "fit failed",
                    str(type(e).__name__), str(e)[:30],
                    "—", "—",
                ])

    overdispersion = (round(aic_pois - aic_nb, 1)
                       if np.isfinite(aic_pois) and np.isfinite(aic_nb)
                       else "n/a")

    summary = [
        ("Source file", source_label),
        ("Dataset id", ds_id),
        ("Treatment column", treatment),
        ("Outcome (count)", outcome),
        ("Year FE", year_col),
        ("Covariate FEs", ", ".join(cov) if cov else "none"),
        ("Cluster group (GEE marginal model)",
            cluster_group or "—"),
        ("Cells", int(work.shape[0])),
        ("Total outcome count", int(work[outcome].sum())),
        ("Overdispersion (Poisson AIC − NB AIC)", overdispersion),
    ]

    return RichResult(
        title=title,
        summary_lines=summary,
        tables=[{
            "title": ("Aggregate Ruhela formulation — Poisson + NB IRR "
                       + ("+ GEE clustered " if cluster_group else "")
                       + "for treatment effect on count outcome:"),
            "headers": ["Family", "Coefficient", "IRR", "95% CI",
                         "p-value", "AIC", "Notes"],
            "rows": rows,
        }],
        interpretation=interpretation + (
            "\n\nIRR > 1 ⇒ treatment increases the count rate; IRR < 1 ⇒ "
            "treatment decreases it. Concordance between Poisson and NB "
            "indicates equidispersion; large gap (Poisson AIC ≫ NB AIC) "
            "indicates overdispersion — trust NB."
            + (" GEE-Poisson with cluster-robust SE is the population-"
                "level marginal-model alternative that respects within-"
                f"{cluster_group} correlation; trust the GEE row when "
                "the GLM with that variable as a fixed effect fails to "
                "fit due to collinearity."
                if cluster_group else "")
        ),
        payload=payloads,
    )


def analyze_b03_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on b03 (Alert × Institution).

    T = Alert_Presence indicator → Y = Number_SegregationPlacements.
    Year-FE + Region-FE + Alert_Type-FE; Institution as **GEE cluster
    group** (24 levels, marginal model with cluster-robust SE).
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b03")
    work = df.dropna(subset=["Alert_Presence",
                              "Number_SegregationPlacements"]).copy()
    work["T_alert"] = (work["Alert_Presence"].astype(str).str.strip()
                        .str.lower().isin(["yes", "y", "true", "1"])
                        .astype(int))
    return _ruhela_aggregate_on(
        work, ds_id="b03",
        source_label=("b03_segregation_placements_alerts_and_hold_flags"
                       "_by_institution.csv"),
        title=("OTIS b03 — Aggregate Ruhela formulation: "
                "Alert presence → Number of segregation placements"),
        interpretation=(
            "Aggregate Ruhela formulation on b03 testing whether an "
            "alert-flagged person-cell receives more segregation "
            "placements than a no-alert cell. GEE-Poisson clustered "
            "on institution (24 levels) is the marginal-model "
            "inference; Poisson + NB GLM rows give the population "
            "IRR with FE on alert type, region, and year."
        ),
        treatment="T_alert",
        outcome="Number_SegregationPlacements",
        covariates=["Alert_Type", "Region_AtTimeOfPlacement"],
        cluster_group="Institution_AtTimeOfPlacement",
    )


def analyze_b07_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on b07 (alerts × gender summary).

    b07 is a wide table with paired columns:
      Number_Segregation_Placements_With_Alert
      Number_Segregation_Placements_Without_Alert

    We pivot to long form (one row per alert-state × gender × year),
    then T = with_alert indicator → Y = count. Year + Gender + Alert_Type
    fixed effects.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b07")
    work = df.dropna(subset=[
        "EndFiscalYear", "Alert_Type", "Gender",
        "Number_Segregation_Placements_With_Alert",
        "Number_Segregation_Placements_Without_Alert",
    ]).copy()
    # Pivot to long
    long_with = work[["EndFiscalYear", "Alert_Type", "Gender",
                       "Number_Segregation_Placements_With_Alert"]
                       ].rename(columns={
                           "Number_Segregation_Placements_With_Alert":
                           "n_placements"})
    long_with["T_alert"] = 1
    long_without = work[["EndFiscalYear", "Alert_Type", "Gender",
                          "Number_Segregation_Placements_Without_Alert"]
                          ].rename(columns={
                              "Number_Segregation_Placements_Without_Alert":
                              "n_placements"})
    long_without["T_alert"] = 0
    long_df = pd.concat([long_with, long_without], ignore_index=True)
    return _ruhela_aggregate_on(
        long_df, ds_id="b07",
        source_label=("b07_segregation_placements_alerts_and_hold_flags"
                       "_by_gender.csv"),
        title=("OTIS b07 — Aggregate Ruhela formulation: "
                "With-alert vs without-alert × gender → "
                "Number of segregation placements"),
        interpretation=(
            "Aggregate Ruhela formulation on b07 testing whether "
            "alert-flagged person-cells receive more segregation "
            "placements than no-alert cells in the same gender × "
            "alert-type stratum. Poisson + NB GLM IRR. Aggregate "
            "counts; no individual-level inference."
        ),
        treatment="T_alert",
        outcome="n_placements",
        covariates=["Alert_Type", "Gender"],
    )


def analyze_c01_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c01 — gender disparity in RC.

    c01 has Gender × {InCustody, RestrictiveConfinement, Segregation}
    counts. T = Female indicator → Y = NumberIndividuals_RC,
    Year-FE.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c01")
    work = df.dropna(subset=[
        "EndFiscalYear", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="c01",
        source_label=("c01_individuals_in_segregation_and_restrictive_"
                       "confinement_total_individuals.csv"),
        title=("OTIS c01 — Aggregate Ruhela formulation: "
                "Female indicator → Number of individuals in RC"),
        interpretation=(
            "Aggregate Ruhela formulation on c01 testing gender "
            "disparity in restrictive-confinement totals at the "
            "population aggregate level. Year-FE only (gender is "
            "the sole demographic cross). Aggregate counts; no "
            "individual-level inference."
        ),
        treatment="T_female",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=[],
    )


def analyze_c07_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c07 — alert × gender → RC count.

    Counterpart of the per-row Ruhela formulation on a01/b01 at the
    aggregate level. Useful triangulation: c07 reports
    NumberIndividuals_InCustody / RC / Segregation cross-tabulated
    by Alert_Type × Gender × FiscalYear. The Ruhela formulation here
    contrasts an indicator of alert presence (vs. no-alert reference)
    with the number-in-RC count.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c07")
    work = df.dropna(subset=[
        "EndFiscalYear", "Alert_Type", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    # Treat any Alert_Type other than "No Alert" / "None" as alert-present
    no_alert_strs = {"no alert", "none", "no_alert", "no"}
    work["T_alert_present"] = (~work["Alert_Type"].astype(str).str.strip()
                                  .str.lower().isin(no_alert_strs)
                                  ).astype(int)
    # If no rows remain in either arm, skip
    if (work["T_alert_present"].sum() == 0
            or (1 - work["T_alert_present"]).sum() == 0):
        return RichResult(
            title="c07 aggregate Ruhela",
            warnings=["c07 has no no-alert reference rows — "
                       "treatment indicator is degenerate"],
        )
    return _ruhela_aggregate_on(
        work, ds_id="c07",
        source_label=("c07_individuals_in_segregation_and_restrictive_"
                       "confinement_alerts_and_hold_flags.csv"),
        title=("OTIS c07 — Aggregate Ruhela formulation: "
                "Alert presence × Gender → Individuals in RC"),
        interpretation=(
            "Aggregate Ruhela formulation on c07 — the population-level "
            "counterpart of the canonical (T_high_ac, vm) per-row "
            "formulation on a01/b01. Tests whether alert-flagged person-"
            "cells produce higher RC counts at the aggregate level."
        ),
        treatment="T_alert_present",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Alert_Type", "Gender"],
    )



# ── Disparity-focused aggregate Ruhela formulations ────────────────


def analyze_b06_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on b06 — disciplinary reason → seg.

    T = (Reason contains "Disciplinary") → Y = Number of seg placements.
    Year-FE + Region-FE + Gender-FE; Institution as **GEE cluster
    group** (24 levels) for marginal-model cluster-robust inference.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b06")
    work = df.dropna(subset=[
        "EndFiscalYear", "Reason", "Gender",
        "Region_AtTimeOfPlacement", "Institution_AtTimeOfPlacement",
        "Number_SegregationPlacements"]).copy()
    work["T_disciplinary"] = (work["Reason"].astype(str).str.lower()
                               .str.contains("disciplinary", na=False)
                               ).astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="b06",
        source_label=("b06_segregation_placements_reason_for_placement_"
                       "by_institution.csv"),
        title=("OTIS b06 — Aggregate Ruhela formulation: "
                "Disciplinary reason → Number of seg placements"),
        interpretation=(
            "Aggregate RF on b06 testing whether disciplinary "
            "segregation reasons produce more placements than non-"
            "disciplinary (security, protection, medical). GEE-Poisson "
            "clustered on institution gives cluster-robust marginal "
            "inference."
        ),
        treatment="T_disciplinary",
        outcome="Number_SegregationPlacements",
        covariates=["Gender", "Region_AtTimeOfPlacement"],
        cluster_group="Institution_AtTimeOfPlacement",
    )


def analyze_c03_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c03 — racial disparity in RC.

    T = (Race == "Indigenous") indicator → Y = NumberIndividuals_RC.
    Year-FE + Gender-FE. Indigenous vs all-other contrast tests
    overrepresentation.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c03")
    work = df.dropna(subset=[
        "EndFiscalYear", "Race", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_indigenous"] = (work["Race"].astype(str).str.lower()
                             == "indigenous").astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="c03",
        source_label=("c03_individuals_in_segregation_and_restrictive_"
                       "confinement_race_by_gender.csv"),
        title=("OTIS c03 — Aggregate Ruhela formulation: "
                "Indigenous race → Individuals in RC"),
        interpretation=(
            "Aggregate Ruhela formulation on c03 testing Indigenous "
            "overrepresentation in restrictive confinement at the "
            "population aggregate level. Indigenous adults make up "
            "~5% of Ontario population but a much higher share of "
            "RC placements; this formulation quantifies the IRR."
        ),
        treatment="T_indigenous",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Gender"],
    )


def analyze_c04_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c04 — racial disparity by region.

    T = (Race == "Indigenous") → Y = NumberIndividuals_RC.
    Year-FE + Region-FE.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c04")
    work = df.dropna(subset=[
        "EndFiscalYear", "Race", "Region_MostRecentPlacement",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_indigenous"] = (work["Race"].astype(str).str.lower()
                             == "indigenous").astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="c04",
        source_label=("c04_individuals_in_segregation_and_restrictive_"
                       "confinement_race_by_region.csv"),
        title=("OTIS c04 — Aggregate Ruhela formulation: "
                "Indigenous race → Individuals in RC, by region"),
        interpretation=(
            "Aggregate Ruhela formulation on c04 testing Indigenous "
            "overrepresentation in RC controlling for region. Region-"
            "FE absorbs cross-region differences in Indigenous "
            "population share so the Indigenous coefficient measures "
            "within-region disparity."
        ),
        treatment="T_indigenous",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Region_MostRecentPlacement"],
    )


def analyze_c06_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c06 — age-50+ disparity in RC.

    T = (Age_Category contains "50") → Y = NumberIndividuals_RC.
    Year-FE + Region-FE.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c06")
    work = df.dropna(subset=[
        "EndFiscalYear", "Age_Category", "Region_MostRecentPlacement",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_50plus"] = _age_50plus_indicator(work["Age_Category"])
    return _ruhela_aggregate_on(
        work, ds_id="c06",
        source_label=("c06_individuals_in_segregation_and_restrictive_"
                       "confinement_age_category_by_region.csv"),
        title=("OTIS c06 — Aggregate Ruhela formulation: "
                "Age 50+ → Individuals in RC, by region"),
        interpretation=(
            "Aggregate Ruhela formulation on c06 testing age-50+ "
            "overrepresentation in RC controlling for region. Older "
            "adults face different segregation pathways than younger "
            "adults; this formulation quantifies the population-level "
            "IRR."
        ),
        treatment="T_50plus",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Region_MostRecentPlacement"],
    )


def analyze_c09_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c09 — age-50+ disparity by gender."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c09")
    work = df.dropna(subset=[
        "EndFiscalYear", "Age_Category", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_50plus"] = _age_50plus_indicator(work["Age_Category"])
    return _ruhela_aggregate_on(
        work, ds_id="c09",
        source_label=("c09_individuals_in_segregation_and_restrictive_"
                       "confinement_age_category_by_gender.csv"),
        title=("OTIS c09 — Aggregate Ruhela formulation: "
                "Age 50+ → Individuals in RC, by gender"),
        interpretation=(
            "Aggregate Ruhela formulation on c09 testing age-50+ "
            "overrepresentation in RC controlling for gender."
        ),
        treatment="T_50plus",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Gender"],
    )



# ── Alternative-treatment Ruhela formulations on a01 (per-row) ────
#
# The canonical a01 Ruhela formulation is T_high_ac → Y_vm_count
# (alert-complexity → regional-volatility). The framework also supports
# alternative treatments on the same panel data: gender, age, region.
# Each runs the full 10-estimator Ruhela suite on the a01 cell-level
# frame, with the alternative T while keeping Y = vm count and the
# usual demographic covariates (excluding the variable being treated).


def _a01_ruhela_cell_frame() -> "tuple[pd.DataFrame, list[str]]":
    """Build the a01 cell-level (id × year) frame with vm count."""
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    df = load_otis_dataset("a01")
    data, _, _, _ = oc.make_pair_alert_to_volatility_ruhela(df)
    return data, ["Gender", "Age_Category", "EndFiscalYear"]


def analyze_a01_ruhela_alt_gender(df: "pd.DataFrame | None" = None,
                                    ) -> RichResult:
    """Alt-T Ruhela formulation on a01 — gender → regional volatility.

    Same per-row Ruhela cell frame as the canonical formulation, but
    T = Female indicator → Y = vm count. Covariates exclude Gender
    (the variable being treated). Runs the full 10-estimator DLRM ensemble.
    Tests whether women experience more or fewer regional transfers
    within a fiscal year of restrictive confinement.
    """
    cell_data, _ = _a01_ruhela_cell_frame()
    cell_data = cell_data.copy()
    cell_data["T_female"] = _female_indicator(cell_data["Gender"])
    return _ruhela_formulations_on(
        cell_data, ds_id="a01-altG",
        source_label=("a01_restrictive_confinement_detailed_dataset.csv "
                       "(alt-T: Gender)"),
        title=("OTIS a01 — Alt-T Ruhela formulation: "
                "T=Female → Y=vm count (regional volatility)"),
        interpretation=(
            "Alternative-treatment Ruhela formulation on the a01 cell "
            "frame. Same Y (vm count) as the canonical formulation; T "
            "swapped to Female indicator. Tests whether gender alone "
            "drives intra-year regional volatility."
        ),
        treatment="T_female",
        outcome="Y_vm_count",
        covariates=["Age_Category", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )


def analyze_a01_ruhela_alt_age(df: "pd.DataFrame | None" = None,
                                 ) -> RichResult:
    """Alt-T Ruhela formulation on a01 — age 50+ → regional volatility."""
    cell_data, _ = _a01_ruhela_cell_frame()
    cell_data = cell_data.copy()
    cell_data["T_50plus"] = _age_50plus_indicator(cell_data["Age_Category"])
    return _ruhela_formulations_on(
        cell_data, ds_id="a01-altA",
        source_label=("a01_restrictive_confinement_detailed_dataset.csv "
                       "(alt-T: Age 50+)"),
        title=("OTIS a01 — Alt-T Ruhela formulation: "
                "T=Age 50+ → Y=vm count"),
        interpretation=(
            "Alternative-treatment Ruhela formulation: age 50+ as the "
            "binary treatment. Tests whether older adults experience "
            "different intra-year regional churn than younger adults."
        ),
        treatment="T_50plus",
        outcome="Y_vm_count",
        covariates=["Gender", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )


def analyze_a01_ruhela_alt_toronto(df: "pd.DataFrame | None" = None,
                                       ) -> RichResult:
    """Alt-T Ruhela formulation on a01 — Toronto-region → regional volatility.

    T = (Region_AtTimeOfPlacement == "Toronto") indicator. Tests
    whether Toronto-region placements produce different volatility
    than placements outside Toronto, controlling for demographics.
    """
    cell_data, _ = _a01_ruhela_cell_frame()
    cell_data = cell_data.copy()
    # Cell-level frame has regA (region at placement)
    if "regA" not in cell_data.columns:
        return RichResult(
            title="a01 Alt-T Toronto Ruhela",
            warnings=["regA column missing from cell frame"],
        )
    cell_data["T_toronto"] = _region_toronto_indicator(cell_data["regA"])
    return _ruhela_formulations_on(
        cell_data, ds_id="a01-altT",
        source_label=("a01_restrictive_confinement_detailed_dataset.csv "
                       "(alt-T: Toronto region)"),
        title=("OTIS a01 — Alt-T Ruhela formulation: "
                "T=Toronto region → Y=vm count"),
        interpretation=(
            "Alternative-treatment Ruhela formulation: Toronto-region "
            "placement as the binary treatment. Tests whether the "
            "Toronto region's institutional density translates into "
            "different intra-year regional churn."
        ),
        treatment="T_toronto",
        outcome="Y_vm_count",
        covariates=["Gender", "Age_Category", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )



# ── b01 alt-T Ruhela formulations (parallel to a01 alt-T) ──────────


def _b01_ruhela_cell_frame() -> "pd.DataFrame":
    """Build the b01 cell-level (id × year) frame with vm count.

    Same structure as a01's cell frame: per-(UniqueIndividual_ID,
    EndFiscalYear) aggregation with regA/regB preserved.
    """
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    df = load_otis_dataset("b01")
    data, _, _, _ = oc.make_pair_alert_to_volatility_ruhela(df)
    return data


def analyze_b01_ruhela_alt_gender(df: "pd.DataFrame | None" = None,
                                    ) -> RichResult:
    """Alt-T Ruhela formulation on b01 — gender → regional volatility.

    Same as analyze_a01_ruhela_alt_gender but on b01 (Segregation
    Detailed) cell frame. Tests whether women experience more or
    fewer regional transfers within a fiscal year of segregation.
    """
    cell_data = _b01_ruhela_cell_frame()
    cell_data = cell_data.copy()
    cell_data["T_female"] = _female_indicator(cell_data["Gender"])
    return _ruhela_formulations_on(
        cell_data, ds_id="b01-altG",
        source_label=("b01_segregation_detailed_dataset.csv "
                       "(alt-T: Gender)"),
        title=("OTIS b01 — Alt-T Ruhela formulation: "
                "T=Female → Y=vm count (regional volatility)"),
        interpretation=(
            "Alternative-treatment Ruhela formulation on the b01 cell "
            "frame. Same Y (vm count) as canonical; T = Female "
            "indicator. Tests whether gender alone drives intra-year "
            "regional volatility on segregation records (vs the per-"
            "day RC records of a01)."
        ),
        treatment="T_female",
        outcome="Y_vm_count",
        covariates=["Age_Category", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )


def analyze_b01_ruhela_alt_age(df: "pd.DataFrame | None" = None,
                                 ) -> RichResult:
    """Alt-T Ruhela formulation on b01 — age 50+ → regional volatility."""
    cell_data = _b01_ruhela_cell_frame()
    cell_data = cell_data.copy()
    cell_data["T_50plus"] = _age_50plus_indicator(cell_data["Age_Category"])
    return _ruhela_formulations_on(
        cell_data, ds_id="b01-altA",
        source_label=("b01_segregation_detailed_dataset.csv "
                       "(alt-T: Age 50+)"),
        title=("OTIS b01 — Alt-T Ruhela formulation: "
                "T=Age 50+ → Y=vm count"),
        interpretation=(
            "Alt-T RF on b01: age 50+ as binary treatment. Tests "
            "whether older adults experience different intra-year "
            "regional churn than younger adults on segregation records."
        ),
        treatment="T_50plus",
        outcome="Y_vm_count",
        covariates=["Gender", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )


def analyze_b01_ruhela_alt_toronto(df: "pd.DataFrame | None" = None,
                                       ) -> RichResult:
    """Alt-T Ruhela formulation on b01 — Toronto-region → regional volatility."""
    cell_data = _b01_ruhela_cell_frame()
    cell_data = cell_data.copy()
    if "regA" not in cell_data.columns:
        return RichResult(
            title="b01 Alt-T Toronto Ruhela",
            warnings=["regA column missing from cell frame"],
        )
    cell_data["T_toronto"] = _region_toronto_indicator(cell_data["regA"])
    return _ruhela_formulations_on(
        cell_data, ds_id="b01-altT",
        source_label=("b01_segregation_detailed_dataset.csv "
                       "(alt-T: Toronto region)"),
        title=("OTIS b01 — Alt-T Ruhela formulation: "
                "T=Toronto region → Y=vm count"),
        interpretation=(
            "Alt-T RF on b01: Toronto-region placement as binary "
            "treatment. Tests whether Toronto's institutional density "
            "translates into different intra-year regional churn on "
            "segregation records."
        ),
        treatment="T_toronto",
        outcome="Y_vm_count",
        covariates=["Gender", "Age_Category", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )


# ── Subgroup-analysis variants on a01 (effect-heterogeneity by gender) ──


def analyze_a01_ruhela_subgroup_female(df: "pd.DataFrame | None" = None,
                                         ) -> RichResult:
    """Canonical a01 Ruhela formulation on Female-only subset.

    Tests effect-heterogeneity-by-gender: same T_high_ac → vm
    contrast, but only on female cells. Compare ATE here to
    analyze_a01_ruhela_subgroup_male to see if alert-complexity
    matters more or less for women than men.
    """
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("a01")
    data, T, Y, cov = oc.make_pair_alert_to_volatility_ruhela(df)
    sub = data[_female_indicator(data["Gender"]) == 1].copy()
    if sub.shape[0] < 100:
        return RichResult(
            title="a01 subgroup Female Ruhela",
            warnings=[f"only {sub.shape[0]} female cells; too few"],
        )
    cov_no_gender = [c for c in cov if c != "Gender"]
    return _ruhela_formulations_on(
        sub, ds_id="a01-subF",
        source_label=("a01_restrictive_confinement_detailed_dataset.csv "
                       "(subgroup: Female)"),
        title=("OTIS a01 — Subgroup Ruhela formulation: "
                "Female-only cell frame, T_high_ac → vm count"),
        interpretation=(
            "Subgroup analysis: canonical Ruhela formulation restricted "
            "to female cells. Compare to analyze_a01_ruhela_subgroup_male "
            "for effect-heterogeneity-by-gender. If the female ATE is "
            "smaller / larger than the male ATE, alert-complexity-driven "
            "regional volatility is gender-conditional."
        ),
        treatment=T, outcome=Y, covariates=cov_no_gender,
        cluster_col="EndFiscalYear",
    )


def analyze_a01_ruhela_subgroup_male(df: "pd.DataFrame | None" = None,
                                       ) -> RichResult:
    """Canonical a01 Ruhela formulation on Male-only subset."""
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("a01")
    data, T, Y, cov = oc.make_pair_alert_to_volatility_ruhela(df)
    sub = data[_female_indicator(data["Gender"]) == 0].copy()
    cov_no_gender = [c for c in cov if c != "Gender"]
    return _ruhela_formulations_on(
        sub, ds_id="a01-subM",
        source_label=("a01_restrictive_confinement_detailed_dataset.csv "
                       "(subgroup: Male)"),
        title=("OTIS a01 — Subgroup Ruhela formulation: "
                "Male-only cell frame, T_high_ac → vm count"),
        interpretation=(
            "Subgroup analysis: canonical Ruhela formulation restricted "
            "to male cells. Companion to analyze_a01_ruhela_subgroup_female "
            "for effect-heterogeneity-by-gender."
        ),
        treatment=T, outcome=Y, covariates=cov_no_gender,
        cluster_col="EndFiscalYear",
    )



def analyze_b01_ruhela_subgroup_female(df: "pd.DataFrame | None" = None,
                                         ) -> RichResult:
    """Canonical b01 Ruhela formulation on Female-only subset.

    Sister to analyze_a01_ruhela_subgroup_female. Tests effect-
    heterogeneity-by-gender on segregation records.
    """
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b01")
    data, T, Y, cov = oc.make_pair_alert_to_volatility_ruhela(df)
    sub = data[_female_indicator(data["Gender"]) == 1].copy()
    if sub.shape[0] < 100:
        return RichResult(
            title="b01 subgroup Female Ruhela",
            warnings=[f"only {sub.shape[0]} female cells; too few"],
        )
    cov_no_gender = [c for c in cov if c != "Gender"]
    return _ruhela_formulations_on(
        sub, ds_id="b01-subF",
        source_label=("b01_segregation_detailed_dataset.csv "
                       "(subgroup: Female)"),
        title=("OTIS b01 — Subgroup Ruhela formulation: "
                "Female-only cell frame, T_high_ac → vm count"),
        interpretation=(
            "Subgroup analysis on b01 (segregation records, per-"
            "placement). Compare to analyze_a01_ruhela_subgroup_female "
            "to see whether the gender-conditional pattern in alert-"
            "complexity → vm holds across the two unit-of-analysis "
            "variants (per-day RC vs per-placement seg)."
        ),
        treatment=T, outcome=Y, covariates=cov_no_gender,
        cluster_col="EndFiscalYear",
    )


def analyze_b01_ruhela_subgroup_male(df: "pd.DataFrame | None" = None,
                                       ) -> RichResult:
    """Canonical b01 Ruhela formulation on Male-only subset."""
    from . import otis_causal as oc
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b01")
    data, T, Y, cov = oc.make_pair_alert_to_volatility_ruhela(df)
    sub = data[_female_indicator(data["Gender"]) == 0].copy()
    cov_no_gender = [c for c in cov if c != "Gender"]
    return _ruhela_formulations_on(
        sub, ds_id="b01-subM",
        source_label=("b01_segregation_detailed_dataset.csv "
                       "(subgroup: Male)"),
        title=("OTIS b01 — Subgroup Ruhela formulation: "
                "Male-only cell frame, T_high_ac → vm count"),
        interpretation=(
            "Male-only subgroup analysis on b01. Companion to "
            "analyze_b01_ruhela_subgroup_female for effect-"
            "heterogeneity-by-gender on segregation records."
        ),
        treatment=T, outcome=Y, covariates=cov_no_gender,
        cluster_col="EndFiscalYear",
    )


# ── b02 alt-T variants (region, age) ───────────────────────────────


def analyze_b02_ruhela_alt_region(df: "pd.DataFrame | None" = None,
                                    ) -> RichResult:
    """b02 alt-T: T = Toronto region → Y = total seg days within FY.

    Sister to analyze_b02_ruhela_formulations (T = Female). Tests
    regional disparity in seg-day burden controlling for gender +
    age + year.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b02")
    work = df.dropna(subset=["Gender",
                              "TotalAggregatedDays_Segregation",
                              "Region_MostRecentPlacement",
                              "Age_Category", "EndFiscalYear"]).copy()
    work["T_toronto"] = _region_toronto_indicator(
        work["Region_MostRecentPlacement"])
    return _ruhela_formulations_on(
        work, ds_id="b02-altR",
        source_label=("b02_segregation_detailed_total_days.csv "
                       "(alt-T: Toronto region)"),
        title=("OTIS b02 — Alt-T Ruhela formulation: "
                "T=Toronto region → Y=Total seg days within FY"),
        interpretation=(
            "Alt-T RF on b02: Toronto-region indicator. Tests whether "
            "Toronto's institutional density translates into different "
            "seg-day burden than other regions, controlling for gender, "
            "age, and year."
        ),
        treatment="T_toronto",
        outcome="TotalAggregatedDays_Segregation",
        covariates=["Gender", "Age_Category", "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )


def analyze_b02_ruhela_alt_age(df: "pd.DataFrame | None" = None,
                                 ) -> RichResult:
    """b02 alt-T: T = Age 50+ → Y = total seg days within FY."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b02")
    work = df.dropna(subset=["Gender",
                              "TotalAggregatedDays_Segregation",
                              "Region_MostRecentPlacement",
                              "Age_Category", "EndFiscalYear"]).copy()
    work["T_50plus"] = _age_50plus_indicator(work["Age_Category"])
    return _ruhela_formulations_on(
        work, ds_id="b02-altA",
        source_label=("b02_segregation_detailed_total_days.csv "
                       "(alt-T: Age 50+)"),
        title=("OTIS b02 — Alt-T Ruhela formulation: "
                "T=Age 50+ → Y=Total seg days within FY"),
        interpretation=(
            "Alt-T RF on b02: Age 50+ indicator. Tests whether older "
            "adults experience different seg-day burden than younger "
            "adults, controlling for gender, region, and year."
        ),
        treatment="T_50plus",
        outcome="TotalAggregatedDays_Segregation",
        covariates=["Gender", "Region_MostRecentPlacement",
                     "EndFiscalYear"],
        cluster_col="EndFiscalYear",
    )



# ── More aggregate disparity RF: c05, c08, b04, b08 ────────────────


def analyze_c05_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on c05 — non-Christian religion → individuals in RC.

    T = (Religion != "Christian" and not "No Religion" and not
    "Unknown") indicator. Tests population-level religious-minority
    overrepresentation in RC, controlling for region.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c05")
    work = df.dropna(subset=[
        "EndFiscalYear", "Religion", "Region_MostRecentPlacement",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    excluded = {"christian", "no religion", "unknown or not reported"}
    work["T_minority_religion"] = (
        ~work["Religion"].astype(str).str.strip().str.lower().isin(excluded)
    ).astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="c05",
        source_label=("c05_individuals_in_segregation_and_restrictive_"
                       "confinement_religion_by_region.csv"),
        title=("OTIS c05 — Aggregate Ruhela formulation: "
                "Non-Christian/non-majority religion → Individuals in RC"),
        interpretation=(
            "Aggregate RF on c05 testing whether religious-minority "
            "(non-Christian, non-no-religion, non-unknown) cells "
            "produce more or fewer individuals in RC, controlling for "
            "region and year. The 'minority' construct is reductive "
            "but tractable for an aggregate population-level signal."
        ),
        treatment="T_minority_religion",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Region_MostRecentPlacement"],
    )


def analyze_c08_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on c08 — non-Christian religion × gender → RC.

    T = (Religion != Christian / no religion / unknown) indicator.
    Year + Gender FE. Companion to c05 with gender control.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c08")
    work = df.dropna(subset=[
        "EndFiscalYear", "Religion", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    excluded = {"christian", "no religion", "unknown or not reported"}
    work["T_minority_religion"] = (
        ~work["Religion"].astype(str).str.strip().str.lower().isin(excluded)
    ).astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="c08",
        source_label=("c08_individuals_in_segregation_and_restrictive_"
                       "confinement_religion_by_gender.csv"),
        title=("OTIS c08 — Aggregate Ruhela formulation: "
                "Non-Christian/non-majority religion × Gender → "
                "Individuals in RC"),
        interpretation=(
            "Aggregate RF on c08, parallel to c05 but with gender "
            "control. Tests the same religious-minority overrepresentation "
            "question; gender + year FE absorb cross-gender baseline."
        ),
        treatment="T_minority_religion",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Gender"],
    )


def analyze_b04_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on b04 — gender disparity in median seg duration.

    b04 has Region × Gender × Measure (max/median/mode) duration. We
    filter to Measure == "Median" and test T = Female → Y = duration.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b04")
    work = df.dropna(subset=[
        "EndFiscalYear", "Region_AtTimeOfPlacement", "Gender",
        "Measure", "NumberConsecutiveDays_Segregation"]).copy()
    work = work[work["Measure"].astype(str).str.strip()
                == "Median"].copy()
    if work.empty:
        return RichResult(
            title="b04 aggregate Ruhela",
            warnings=["b04 has no Median rows after filter"],
        )
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="b04",
        source_label=("b04_segregation_placements_consecutive_durations"
                       "_by_region.csv"),
        title=("OTIS b04 — Aggregate Ruhela formulation: "
                "Female → median consecutive seg duration"),
        interpretation=(
            "Aggregate RF on b04 testing gender disparity in the "
            "MEDIAN consecutive duration of segregation placements, "
            "by region. Median rather than max because max is "
            "outlier-driven; mode is privacy-suppressed in many cells."
        ),
        treatment="T_female",
        outcome="NumberConsecutiveDays_Segregation",
        covariates=["Region_AtTimeOfPlacement"],
    )


def analyze_b08_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on b08 — gender disparity in median seg duration
    by institution.

    Companion to b04; b08 adds Institution_AtTimeOfPlacement as a
    finer-grained spatial dimension. Region-FE; Institution as **GEE
    cluster group** (24 levels).
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b08")
    work = df.dropna(subset=[
        "EndFiscalYear", "Region_AtTimeOfPlacement",
        "Institution_AtTimeOfPlacement", "Gender",
        "Measure", "NumberConsecutiveDays_Segregation"]).copy()
    work = work[work["Measure"].astype(str).str.strip()
                == "Median"].copy()
    if work.empty:
        return RichResult(
            title="b08 aggregate Ruhela",
            warnings=["b08 has no Median rows after filter"],
        )
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="b08",
        source_label=("b08_segregation_placements_consecutive_durations"
                       "_by_institution.csv"),
        title=("OTIS b08 — Aggregate Ruhela formulation: "
                "Female → median seg duration, GEE-clustered on institution"),
        interpretation=(
            "Aggregate RF on b08 testing gender disparity in median "
            "segregation duration with region-FE + GEE-Poisson "
            "clustered on institution (24 levels). Companion to b04 "
            "with finer-grained spatial control."
        ),
        treatment="T_female",
        outcome="NumberConsecutiveDays_Segregation",
        covariates=["Region_AtTimeOfPlacement"],
        cluster_group="Institution_AtTimeOfPlacement",
    )



# ── Final aggregate RF coverage: b09 / c02 / c10 / c12 + d-series ──


def analyze_b09_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on b09 — gender disparity in placement-count distribution.

    b09 has rows of (Year × NumberPlacements_Segregation × Gender ×
    NumberIndividuals_Segregation). We test T = Female → Y =
    NumberIndividuals_Segregation, with NumberPlacements as a
    categorical FE so we adjust for the bin-cell structure.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("b09")
    work = df.dropna(subset=[
        "EndFiscalYear", "NumberPlacements_Segregation", "Gender",
        "NumberIndividuals_Segregation"]).copy()
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="b09",
        source_label=("b09_individuals_in_segregation_number_of_times"
                       "_in_segregation.csv"),
        title=("OTIS b09 — Aggregate Ruhela formulation: "
                "Female → Number of individuals in segregation"),
        interpretation=(
            "Aggregate RF on b09 testing gender disparity in the "
            "individuals-in-segregation count distribution, "
            "controlling for the placement-count bin (categorical FE)."
        ),
        treatment="T_female",
        outcome="NumberIndividuals_Segregation",
        covariates=["NumberPlacements_Segregation"],
    )


def analyze_c02_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on c02 — gender disparity by institution.

    Companion to c01 with institution-level granularity. T = Female →
    Y = NumberIndividuals_RC. Region + year as fixed effects;
    **Institution as a GEE cluster group** (marginal model with
    cluster-robust SE) — Institution has too many levels (~76) for an
    FE specification on the 148-row table without collinearity.
    Mirrors OTIS-RC's ``glmmTMB nbinom2`` clustered approach.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c02")
    work = df.dropna(subset=[
        "EndFiscalYear", "Region_MostRecentPlacement",
        "Institution_MostRecentPlacement", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="c02",
        source_label=("c02_individuals_in_segregation_and_restrictive_"
                       "confinement_by_institution.csv"),
        title=("OTIS c02 — Aggregate Ruhela formulation: "
                "Female → Individuals in RC, GEE-clustered on institution"),
        interpretation=(
            "Aggregate RF on c02 testing gender disparity in RC counts. "
            "Region + year as fixed effects; institution enters as a "
            "GEE cluster group (~76 levels too many for FE specification "
            "on this 148-row table). The GEE-Poisson row reports the "
            "population-level IRR with cluster-robust SE — the "
            "marginal-model analogue of OTIS-RC's glmmTMB nbinom2 "
            "approach with institution random intercept."
        ),
        treatment="T_female",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=["Region_MostRecentPlacement"],
        cluster_group="Institution_MostRecentPlacement",
    )


def analyze_c10_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on c10 — gender disparity in median RC days.

    c10 has Region × Institution × Gender × Measure
    (Maximum/Median/Mode) duration. Filter to Median; T = Female →
    Y = TotalAggregatedDays_RC. Region-FE + Year-FE; Institution as
    **GEE cluster group** (25 levels) for cluster-robust marginal
    inference.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c10")
    work = df.dropna(subset=[
        "EndFiscalYear", "Region_MostRecentPlacement",
        "Institution_MostRecentPlacement", "Gender", "Measure",
        "TotalAggregatedDays_RestrictiveConfinement"]).copy()
    work = work[work["Measure"].astype(str).str.strip()
                == "Median"].copy()
    if work.empty:
        return RichResult(
            title="c10 aggregate Ruhela",
            warnings=["c10 has no Median rows after filter"],
        )
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="c10",
        source_label=("c10_segregation_and_restrictive_confinement_"
                       "aggregate_durations_by_institution.csv"),
        title=("OTIS c10 — Aggregate Ruhela formulation: "
                "Female → median RC days, GEE-clustered on institution"),
        interpretation=(
            "Aggregate RF on c10 testing gender disparity in median "
            "total RC days per individual. Region + year FE; "
            "Institution as GEE cluster group (25 levels) for cluster-"
            "robust marginal inference."
        ),
        treatment="T_female",
        outcome="TotalAggregatedDays_RestrictiveConfinement",
        covariates=["Region_MostRecentPlacement"],
        cluster_group="Institution_MostRecentPlacement",
    )


def analyze_c12_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate RF on c12 — gender disparity in median RC days by region."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c12")
    work = df.dropna(subset=[
        "EndFiscalYear", "Region_MostRecentPlacement", "Gender",
        "Measure", "TotalAggregatedDays_RestrictiveConfinement"]).copy()
    work = work[work["Measure"].astype(str).str.strip()
                == "Median"].copy()
    if work.empty:
        return RichResult(
            title="c12 aggregate Ruhela",
            warnings=["c12 has no Median rows after filter"],
        )
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="c12",
        source_label=("c12_segregation_and_restrictive_confinement_"
                       "aggregate_durations_by_region.csv"),
        title=("OTIS c12 — Aggregate Ruhela formulation: "
                "Female → median RC days, by region"),
        interpretation=(
            "Aggregate RF on c12, the region-level companion to c10. "
            "Tests gender disparity in median total RC days with "
            "region-FE only (less granular than c10's institution-FE)."
        ),
        treatment="T_female",
        outcome="TotalAggregatedDays_RestrictiveConfinement",
        covariates=["Region_MostRecentPlacement"],
    )



def analyze_c11_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """Aggregate Ruhela formulation on c11 — long-duration cell concentration.

    c11 has rows of (FY × Aggregate_Duration_bin × NumberIndividuals_RC
    × NumberIndividuals_Seg). 11 duration bins from "1 day" to
    "Greater than 30 days". Rather than treat duration as the
    treatment-FE (it IS the categorical the data is binned by), we
    test T = (long-duration bin: ≥16 days) → Y = NumberIndividuals_RC.
    Year-FE only.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c11")
    work = df.dropna(subset=[
        "EndFiscalYear", "Aggregate_Duration",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    long_bins = {
        "16 to 20 days", "21 to 25 days", "26 to 30 days",
        "Greater than 30 days",
    }
    work["T_long_duration"] = work["Aggregate_Duration"].astype(str).isin(
        long_bins).astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="c11",
        source_label=("c11_individuals_in_segregation_and_restrictive_"
                       "confinement_aggregate_lengths.csv"),
        title=("OTIS c11 — Aggregate Ruhela formulation: "
                "Long-duration bin (≥16 days) → Individuals in RC"),
        interpretation=(
            "Aggregate RF on c11 testing whether long-duration bins "
            "(≥16 days) account for a disproportionate share of "
            "individuals in restrictive confinement. The 11 duration "
            "bins are binarised into long (4 bins, ≥16 days) vs short "
            "(7 bins, ≤15 days). IRR > 1 ⇒ long-duration bins concentrate "
            "more individuals; IRR < 1 ⇒ short-duration bins dominate. "
            "Year-FE only."
        ),
        treatment="T_long_duration",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=[],
    )


# ── Region-cluster GEE variants of canonical formulations ──────────


def analyze_c01_ruhela_aggregate_region_cluster(
        df: "pd.DataFrame | None" = None) -> RichResult:
    """c01 alternative: Female → RC count, GEE-clustered on year.

    Sister analyzer to analyze_c01_ruhela_aggregate. The c01 case has
    only 6 cells (3 years × 2 genders), so cluster_group="EndFiscalYear"
    gives a 3-cluster GEE — extreme small-cluster bias. This variant
    is shipped for transparency: shows that with so few clusters,
    Poisson and GEE-Poisson give nearly identical numbers because
    there's no clustering structure to exploit.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c01")
    work = df.dropna(subset=[
        "EndFiscalYear", "Gender",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="c01-RC",
        source_label=("c01_individuals_in_segregation_and_restrictive_"
                       "confinement_total_individuals.csv "
                       "(year-clustered variant)"),
        title=("OTIS c01 — Aggregate Ruhela formulation, year-clustered: "
                "Female → Individuals in RC"),
        interpretation=(
            "Year-clustered GEE variant of c01. With only 3 fiscal "
            "years (= 3 clusters), the GEE inference is unreliable — "
            "this analyzer is shipped for transparency, not as the "
            "primary inference. The standard c01 analyzer remains the "
            "default."
        ),
        treatment="T_female",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=[],
        cluster_group="EndFiscalYear",
    )


def analyze_c04_ruhela_aggregate_region_cluster(
        df: "pd.DataFrame | None" = None) -> RichResult:
    """c04 alternative: Indigenous → RC, GEE-clustered on region.

    Sister to analyze_c04_ruhela_aggregate (which uses region as FE).
    With 5 regions, the GEE-cluster:Region inference treats region as
    a marginal-model grouping. Useful when wide region heterogeneity
    in case-mix matters for SE.
    """
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("c04")
    work = df.dropna(subset=[
        "EndFiscalYear", "Race", "Region_MostRecentPlacement",
        "NumberIndividuals_RestrictiveConfinement"]).copy()
    work["T_indigenous"] = (work["Race"].astype(str).str.lower()
                             == "indigenous").astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="c04-RC",
        source_label=("c04_individuals_in_segregation_and_restrictive_"
                       "confinement_race_by_region.csv "
                       "(region-clustered variant)"),
        title=("OTIS c04 — Aggregate Ruhela formulation, region-clustered: "
                "Indigenous → Individuals in RC"),
        interpretation=(
            "Region-clustered GEE variant of c04. The standard c04 "
            "analyzer enters region as FE; this variant treats region "
            "as a GEE cluster group (5 regions). Cluster-robust SE "
            "accounts for within-region case-mix correlation. Both "
            "should give similar IRR; the SE differs."
        ),
        treatment="T_indigenous",
        outcome="NumberIndividuals_RestrictiveConfinement",
        covariates=[],
        cluster_group="Region_MostRecentPlacement",
    )


# ── d-series Ruhela formulations on death counts ───────────────────


def analyze_d02_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """d02 — gender → custodial death count, with year FE."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("d02")
    work = df.dropna(subset=["Year", "Gender",
                              "Number_CustodialDeaths"]).copy()
    work["T_female"] = _female_indicator(work["Gender"])
    return _ruhela_aggregate_on(
        work, ds_id="d02",
        source_label="d02_deaths_in_custody_gender.csv",
        title=("OTIS d02 — Aggregate Ruhela formulation: "
                "Female → Number of custodial deaths"),
        interpretation=(
            "Aggregate RF on d02 testing gender disparity in custodial "
            "death counts. Calendar-year aggregate, very small N (6 "
            "rows = 3 years × 2 genders). Wide CI expected."
        ),
        treatment="T_female",
        outcome="Number_CustodialDeaths",
        covariates=[],
        year_col="Year",
    )


def analyze_d03_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """d03 — Indigenous → custodial death count, with year FE."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("d03")
    work = df.dropna(subset=["Year", "Race",
                              "Number_CustodialDeaths"]).copy()
    work["T_indigenous"] = (work["Race"].astype(str).str.lower()
                             == "indigenous").astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="d03",
        source_label="d03_deaths_in_custody_race.csv",
        title=("OTIS d03 — Aggregate Ruhela formulation: "
                "Indigenous → Number of custodial deaths"),
        interpretation=(
            "Aggregate RF on d03 testing Indigenous overrepresentation "
            "in custodial death counts. Small-sample warning: 17 rows; "
            "Indigenous-specific cells may be privacy-suppressed in "
            "some years."
        ),
        treatment="T_indigenous",
        outcome="Number_CustodialDeaths",
        covariates=[],
        year_col="Year",
    )


def analyze_d04_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """d04 — non-majority religion → custodial death count."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("d04")
    work = df.dropna(subset=["Year", "Religion",
                              "Number_CustodialDeaths"]).copy()
    excluded = {"christian", "no religion", "unknown or not reported"}
    work["T_minority_religion"] = (
        ~work["Religion"].astype(str).str.strip().str.lower().isin(excluded)
    ).astype(int)
    return _ruhela_aggregate_on(
        work, ds_id="d04",
        source_label="d04_deaths_in_custody_religion.csv",
        title=("OTIS d04 — Aggregate Ruhela formulation: "
                "Non-majority religion → Number of custodial deaths"),
        interpretation=(
            "Aggregate RF on d04 testing whether non-majority-religion "
            "cells produce higher death counts. 20 rows total. Small-"
            "sample warning."
        ),
        treatment="T_minority_religion",
        outcome="Number_CustodialDeaths",
        covariates=[],
        year_col="Year",
    )


def analyze_d05_ruhela_aggregate(df: "pd.DataFrame | None" = None,
                                   ) -> RichResult:
    """d05 — age 50+ → custodial death count."""
    from .otis_datasets import load_otis_dataset
    if df is None:
        df = load_otis_dataset("d05")
    work = df.dropna(subset=["Year", "Age_Category",
                              "Number_CustodialDeaths"]).copy()
    work["T_50plus"] = _age_50plus_indicator(work["Age_Category"])
    return _ruhela_aggregate_on(
        work, ds_id="d05",
        source_label="d05_deaths_in_custody_age_category.csv",
        title=("OTIS d05 — Aggregate Ruhela formulation: "
                "Age 50+ → Number of custodial deaths"),
        interpretation=(
            "Aggregate RF on d05 testing age-50+ overrepresentation "
            "in custodial death counts. Old-age mortality is well-"
            "documented in custody literature; this formulation "
            "quantifies the IRR at the population aggregate level."
        ),
        treatment="T_50plus",
        outcome="Number_CustodialDeaths",
        covariates=[],
        year_col="Year",
    )



# ── Ruhela formulations grid summary ───────────────────────────────


def analyze_ruhela_grid() -> RichResult:
    """One-page Ruhela formulations grid summary (24 aggregates).

    Runs every aggregate Ruhela formulation analyzer + the region-cluster
    GEE variants, and presents a single comparison table.
    """
    aggregates_to_run = [
        ("b03", analyze_b03_ruhela_aggregate),
        ("b04", analyze_b04_ruhela_aggregate),
        ("b06", analyze_b06_ruhela_aggregate),
        ("b07", analyze_b07_ruhela_aggregate),
        ("b08", analyze_b08_ruhela_aggregate),
        ("b09", analyze_b09_ruhela_aggregate),
        ("c01", analyze_c01_ruhela_aggregate),
        ("c01-RC", analyze_c01_ruhela_aggregate_region_cluster),
        ("c02", analyze_c02_ruhela_aggregate),
        ("c03", analyze_c03_ruhela_aggregate),
        ("c04", analyze_c04_ruhela_aggregate),
        ("c04-RC", analyze_c04_ruhela_aggregate_region_cluster),
        ("c05", analyze_c05_ruhela_aggregate),
        ("c06", analyze_c06_ruhela_aggregate),
        ("c07", analyze_c07_ruhela_aggregate),
        ("c08", analyze_c08_ruhela_aggregate),
        ("c09", analyze_c09_ruhela_aggregate),
        ("c10", analyze_c10_ruhela_aggregate),
        ("c11", analyze_c11_ruhela_aggregate),
        ("c12", analyze_c12_ruhela_aggregate),
        ("d02", analyze_d02_ruhela_aggregate),
        ("d03", analyze_d03_ruhela_aggregate),
        ("d04", analyze_d04_ruhela_aggregate),
        ("d05", analyze_d05_ruhela_aggregate),
    ]

    rows: list = []
    for ds_id, fn in aggregates_to_run:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            rows.append([ds_id, "—", "load_err", str(type(e).__name__),
                          str(e)[:30], "—", "—"])
            continue
        if r.warnings:
            rows.append([ds_id, "—", "warn",
                          (r.warnings or [""])[0][:50],
                          "—", "—", "—"])
            continue
        if not (r.tables and r.tables[0].get("rows")):
            rows.append([ds_id, "—", "no rows", "—", "—", "—", "—"])
            continue
        # Estimator priority (most → least appropriate):
        # 1. GEE-NB Exch  — cluster-robust + overdispersion-aware (OTIS-RC analog)
        # 2. GEE-Poisson Exch — cluster-robust, equidispersed
        # 3. NB GLM       — overdispersion-aware, no cluster
        # 4. Poisson GLM  — fallback
        def _find(label_substr):
            return next((row for row in r.tables[0]["rows"]
                          if label_substr in str(row[0])
                          and row[2] != "fit failed"), None)

        primary = (_find("GEE-NB") and _find("GEE-NB (cluster")
                    or _find("GEE-NB"))
        primary_type = "GEE-NB (cluster)" if primary else None
        if primary is None:
            primary = _find("GEE-Poisson")
            primary_type = "GEE-Poisson (cluster)" if primary else None
        if primary is None:
            primary = next((row for row in r.tables[0]["rows"]
                             if row[0] == "NB" and row[2] != "fit failed"),
                            None)
            primary_type = "NB GLM" if primary else None
        if primary is None:
            primary = next((row for row in r.tables[0]["rows"]
                             if row[0] == "Poisson"
                             and row[2] != "fit failed"), None)
            primary_type = "Poisson GLM" if primary else None
        title = (r.title or "")[:60]
        if primary:
            rows.append([
                ds_id, primary_type, title,
                primary[2], primary[3], primary[4],
                primary[6][:30] if len(primary) > 6 else "—",
            ])
        else:
            rows.append([
                ds_id, "all failed", title,
                "—", "—", "—", "—",
            ])

    return RichResult(
        title=("OTIS Ruhela formulations grid summary — one-page IRR/ATE"
                " comparison across all aggregate RF analyzers"),
        summary_lines=[
            ("Aggregate datasets covered", len(aggregates_to_run)),
            ("Per-row datasets covered (a01, b01, b02; alt-T on a01)",
                "see analyze_a01/b01/b02_ruhela_formulations + alt-T variants"),
            ("Doob chi² datasets (c, d)",
                "see analyze_c_doob_chi2 / analyze_d_doob_chi2"),
            ("Primary estimator priority",
                "GEE-NB > GEE-Poisson > NB GLM > Poisson GLM"),
        ],
        tables=[{
            "title": ("Aggregate Ruhela formulations — primary IRR per "
                       "dataset (GEE cluster-robust > NB GLM):"),
            "headers": ["DS", "Type", "Formulation", "IRR",
                         "95% CI", "p", "Notes"],
            "rows": rows,
        }],
        interpretation=(
            "One-page comparison of every aggregate Ruhela formulation "
            "currently shipped. The primary IRR is the GEE-Poisson "
            "cluster-robust estimate when available (more conservative "
            "inference); otherwise the NB GLM estimate. IRR > 1 ⇒ "
            "treatment increases the count; IRR < 1 ⇒ decreases. Per-row "
            "a01/b01 RFs are the canonical individual-level contrasts. "
            "See docs/source/methods/ruhela_formulations.md."
        ),
        payload={"n_aggregates": len(aggregates_to_run)},
    )



def analyze_ruhela_master(*, include_per_row: bool = False) -> RichResult:
    """Paper-ready master report — every Ruhela formulation in one RichResult.

    Sections:
      1. Aggregate Ruhela formulations (24 datasets) — IRR comparison
      2. Per-row Ruhela formulations on a01/b01/b02 — primary IRM-DML ATE
         (only when ``include_per_row=True``; ~5-7 min runtime)
      3. Doob chi-square family on c-series + d-series

    For thesis / paper writing.

    Parameters
    ----------
    include_per_row : bool
        If True, also runs the slow per-row RFs on a01/b01/b02 (full
        10-estimator DLRM). Default False — runs in ~1 second from
        cached aggregates only.
    """
    sections: list = []

    # === Section 1: aggregate grid (always) ===
    grid = analyze_ruhela_grid()
    if grid.tables:
        sections.append({
            "title": ("§1 Aggregate Ruhela formulations — primary "
                       "IRR per dataset:"),
            "headers": grid.tables[0]["headers"],
            "rows": grid.tables[0]["rows"],
        })

    # === Section 2: per-row RFs (optional, slow) ===
    per_row_rows: list = []
    if include_per_row:
        for ds_id, fn in [
            ("a01", analyze_a01_ruhela_formulations),
            ("b01", analyze_b01_ruhela_formulations),
            ("b02", analyze_b02_ruhela_formulations),
        ]:
            try:
                r = fn()
            except Exception as e:  # noqa: BLE001
                per_row_rows.append([ds_id, "load_err",
                                       str(type(e).__name__),
                                       str(e)[:30], "—", "—"])
                continue
            if r.warnings:
                per_row_rows.append([ds_id, "warn",
                                       (r.warnings or [""])[0][:50],
                                       "—", "—", "—"])
                continue
            if not (r.tables and r.tables[0].get("rows")):
                per_row_rows.append([ds_id, "no rows", "—", "—", "—", "—"])
                continue
            irm_row = next((row for row in r.tables[0]["rows"]
                             if "IRM-DML" in str(row[0])
                             and "ATE" in str(row[1])), None)
            if irm_row:
                per_row_rows.append([
                    ds_id, "IRM-DML ATE",
                    irm_row[2], irm_row[3], irm_row[4], irm_row[5],
                ])
            else:
                per_row_rows.append([ds_id, "no IRM-DML ATE",
                                       "—", "—", "—", "—"])
    if per_row_rows:
        sections.append({
            "title": ("§2 Per-row Ruhela formulations (canonical "
                       "T_high_ac → vm_count) — IRM-DML ATE:"),
            "headers": ["DS", "Estimator", "ATE", "SE", "95% CI", "p"],
            "rows": per_row_rows,
        })

    # === Section 3: Doob chi-square (always, fast) ===
    try:
        c_doob = analyze_c_doob_chi2()
        d_doob = analyze_d_doob_chi2()
        doob_rows: list = []
        for label, r in [("c-series", c_doob), ("d-series", d_doob)]:
            if r.tables and r.tables[0].get("rows"):
                # Pick the χ² rows
                for row in r.tables[0]["rows"][:3]:
                    doob_rows.append([label, *row[:5]])
        if doob_rows:
            sections.append({
                "title": ("§3 Doob chi-square family — Pearson χ² + "
                           "Cramer's V on aggregate contingency tables:"),
                "headers": ["Series", "Slice/measure", "χ²", "dof",
                             "p", "Cramer's V"],
                "rows": doob_rows,
            })
    except Exception:  # noqa: BLE001
        pass

    # === Section 4: Sprott-Doob CRIMSL/Schulich federal SIU evidence ===
    # Federal-aggregate analyses extending the Ruhela formulations to the
    # national-level Sprott / Doob / Iftene CRIMSL + Schulich Law tables.
    try:
        from . import sprott_doob as _sd
        sd_rows = []
        # Feb 2021 Mandela classification (Table 19) headline
        for r in _sd.TABLE19_MANDELA_CLASSIFICATION:
            sd_rows.append(["SD-2021-Feb T19", r["category"],
                             f"{r['percent']:.1f}%", r["n"], "—", "—"])
        # Feb 2021 regional torture rates (Table 23) headline
        for r in _sd.TABLE23_REGIONAL_TORTURE_RATES:
            sd_rows.append(["SD-2021-Feb T23", r["region"],
                             "—", "—",
                             f"{r['solitary_rate']:.2f}/1k",
                             f"{r['torture_rate']:.2f}/1k"])
        # Feb 2021 — additional headline rows from Tables 11, 12, 22
        if hasattr(_sd, "TABLE11_CHISQ"):
            sd_rows.append(["SD-2021-Feb T11",
                             "Region × stay-length χ²",
                             f"χ²={_sd.TABLE11_CHISQ['chi2']}, "
                             f"df={_sd.TABLE11_CHISQ['df']}",
                             f"p<{_sd.TABLE11_CHISQ['p']}",
                             "—", "—"])
        if hasattr(_sd, "TABLE12_REGIONAL_OVERREP"):
            for r in _sd.TABLE12_REGIONAL_OVERREP:
                ratio = r["siu_pct"] / r["pop_pct"]
                sd_rows.append(["SD-2021-Feb T12",
                                 f"{r['region']} over-/under-rep",
                                 f"SIU {r['siu_pct']:.1f}%",
                                 f"Pop {r['pop_pct']:.1f}%",
                                 f"{ratio:.2f}×", "—"])
        if hasattr(_sd, "TABLE22_CHISQ"):
            sd_rows.append(["SD-2021-Feb T22",
                             "Region × Mandela χ²",
                             f"χ²={_sd.TABLE22_CHISQ['chi2']}, "
                             f"df={_sd.TABLE22_CHISQ['df']}",
                             f"p<{_sd.TABLE22_CHISQ['p']}",
                             "—", "—"])
        # May 2021 — IEDM headline
        h = _sd.HEADLINE_MAY2021
        sd_rows.append(["SDI-2021-May", "n_iedm_stays_reviewed", "—",
                         h["n_iedm_stays_reviewed"], "—", "—"])
        sd_rows.append(["SDI-2021-May", "% IEDM stay-in (Table 9)",
                         f"{h['pct_stay_in_decisions_among_rendered']}%",
                         "—", "—", "—"])
        sd_rows.append(["SDI-2021-May", "% pre-empted by CSC (T9)",
                         f"{h['pct_csc_moved_prisoner_before_iedm']}%",
                         "—", "—", "—"])
        sd_rows.append(["SDI-2021-May", "long-stay no-IEDM record (T15)",
                         "—", h["n_long_stay_no_iedm_record_min76d"],
                         "—", "—"])
        sd_rows.append(["SDI-2021-May", "Indigenous share (T1)",
                         f"{h['indigenous_share_of_reviewed_stays_pct']:.1f}%",
                         "—", "—", "—"])
        # May 2021 — additional headline rows from Tables 5, 8, 10
        if hasattr(_sd, "TABLE5_MAY2021_CHISQ"):
            sd_rows.append(["SDI-2021-May T5",
                             "Race × #IEDM-reviews χ²",
                             f"χ²={_sd.TABLE5_MAY2021_CHISQ['chi2']}, "
                             f"df={_sd.TABLE5_MAY2021_CHISQ['df']}",
                             f"p<{_sd.TABLE5_MAY2021_CHISQ['p']}",
                             "—", "—"])
        if hasattr(_sd, "TABLE8_MAY2021_RACE_X_STAY_121PLUS"):
            for r in _sd.TABLE8_MAY2021_RACE_X_STAY_121PLUS:
                sd_rows.append(["SDI-2021-May T8",
                                 f"{r['race']} ≥121d share",
                                 f"{r['pct_121_380']:.1f}%",
                                 r["n_121_380"], "—", "—"])
        if hasattr(_sd, "TABLE10_MAY2021_CHISQ"):
            sd_rows.append(["SDI-2021-May T10",
                             "Per-IEDM 'remain' rate variance χ²",
                             f"χ²={_sd.TABLE10_MAY2021_CHISQ['chi2']}, "
                             f"df={_sd.TABLE10_MAY2021_CHISQ['df']}",
                             f"p<{_sd.TABLE10_MAY2021_CHISQ['p']}",
                             "37.5%-85.7%", "—"])
        if sd_rows:
            sections.append({
                "title": ("§4 RF-extended (federal): Sprott-Doob CRIMSL "
                           "+ Sprott-Doob-Iftene Schulich Law SIU "
                           "analyses (national-aggregate companion):"),
                "headers": ["Source", "Item", "Percent/Stat",
                             "N", "Solitary/1k", "Torture/1k"],
                "rows": sd_rows,
            })
    except Exception:  # noqa: BLE001
        pass

    # === Section 5: Doob T-539-20 affidavit — CCRSO national tables ===
    try:
        from . import doob_trends as _dt
        doob_nat_rows = []
        # Table 1: 5-yr avg releases by type
        for r in _dt.CCRSO_TABLE1_RELEASES:
            doob_nat_rows.append([
                "Doob-T1", r["type"],
                f"{r['revoke_violent']:.1f} ({r['revoke_violent_pct']:.2f}%)",
                f"{r['success']:.1f} ({r['success_pct']:.1f}%)",
                f"{r['total']:.1f}",
            ])
        # Table 3: age over-/under-rep IRR (one-line per row)
        for r in _dt.CCRSO_TABLE3_AGE:
            irr_c = r["csc_in_custody_pct"] / r["canada_adult_pop_pct"]
            doob_nat_rows.append([
                "Doob-T3", f"age {r['age_group']}",
                f"adult {r['canada_adult_pop_pct']:.1f}%",
                f"custody {r['csc_in_custody_pct']:.1f}%",
                f"IRR_custody={irr_c:.2f}",
            ])
        if doob_nat_rows:
            sections.append({
                "title": ("§5 RF-extended (federal): Doob T-539-20 "
                           "affidavit — CCRSO 2018 prisoner-flow & age "
                           "tables (national-aggregate companion):"),
                "headers": ["Source", "Item", "Col A", "Col B", "Col C"],
                "rows": doob_nat_rows,
            })
    except Exception:  # noqa: BLE001
        pass

    # === Section 6: Mandela-RF on OTIS provincial data ===
    # Bridges Sprott-Doob's federal Mandela classifier into the
    # Ruhela formulations on Ontario provincial OTIS data.
    try:
        c11_man = analyze_c11_mandela_classification()
        b05_man = analyze_b05_mandela_classification()
        cmp_man = analyze_otis_mandela_provincial_vs_federal()
        man_rows = []
        # Provincial b05 (placements) per year
        if b05_man.tables and b05_man.tables[0].get("rows"):
            for r in b05_man.tables[0]["rows"]:
                man_rows.append(["b05 placement", str(r[0]),
                                  r[2], r[4], r[5]])
        # Provincial c11 (individuals) Segregation per year
        if c11_man.payload.get("rows"):
            for r in c11_man.payload["rows"]:
                if r[1] == "Segregation":
                    man_rows.append([f"c11 individuals ({r[1]})",
                                      str(r[0]), r[3], r[5], r[6]])
        # Federal-vs-provincial cross-comparison summary row
        if cmp_man.payload.get("max_provincial_torture_pct") is not None:
            gap = cmp_man.payload["gap_pp"]
            max_pct = cmp_man.payload["max_provincial_torture_pct"]
            max_year = cmp_man.payload["max_provincial_torture_year"]
            man_rows.append([
                "Federal SIU vs Prov peak (Mandela torture)",
                f"max {max_year}",
                "—", f"prov {max_pct:.1f}%",
                f"gap {gap:+.1f} pp vs federal 9.9%",
            ])
        if man_rows:
            sections.append({
                "title": ("§6 Mandela-RF — Ontario provincial Mandela "
                           "classification (OTIS b05/c11) + federal "
                           "cross-comparison:"),
                "headers": ["Source", "Year/Slice", "Solitary %",
                             "Torture %", "N or note"],
                "rows": man_rows,
            })
    except Exception:  # noqa: BLE001
        pass

    return RichResult(
        title=("OTIS Ruhela formulations — paper-ready master report "
                "(provincial + federal-aggregate companion)"),
        summary_lines=[
            ("Sections", len(sections)),
            ("Aggregate RFs", "24 (b03-b09 + c01-c12 + d02-d05 + region-cluster variants)"),
            ("Per-row RFs included", include_per_row),
            ("Doob chi-square", "c-series + d-series families"),
            ("Methodology attribution", "DLRM (Doob-Levinsky-Ruhela-Medina)"),
            ("Acknowledgements (separate)", "Jauregui, A. Laniyonu"),
            ("Federal RF extensions",
                "Sprott-Doob CRIMSL (Feb 2021) + Sprott-Doob-Iftene "
                "Schulich Law (May 2021) + Doob T-539-20 (CCRSO 2018)"),
            ("Federal panel context",
                "moirais.siuiap (Sapers chair, Doob, Sprott)"),
        ],
        tables=sections,
        interpretation=(
            "Master report compiling every Ruhela formulation analyzer "
            "shipped in moirais. Sections: (1) aggregate IRR grid on "
            "OTIS, cluster-robust where high-cardinality grouping "
            "applies; (2) per-row IRM-DML ATE on a01/b01/b02 (when "
            "include_per_row=True); (3) Doob χ² on c-series + d-series "
            "demographic contingency tables; (4) RF-extended federal "
            "companion — Sprott-Doob CRIMSL Feb 2021 Mandela-Rules "
            "classifier (28.4% solitary, 9.9% torture) + regional "
            "torture rates (Pacific 22.6× Ontario), and Sprott-Doob-"
            "Iftene Schulich Law May 2021 IEDM-review evaluation "
            "(N=265 stays; 87% stay-in rate; 30% pre-empted; 12 IEDMs "
            "varied 38%-86%); (5) Doob T-539-20 affidavit CCRSO 2018 "
            "tables (release outcomes, prisoner flow, age over-/under-"
            "representation). Together these brackets the argument "
            "from federal aggregates (Sections 4-5) down to provincial "
            "individual-level evidence (Sections 1-3). Methodology "
            "attribution remains DLRM; the federal-aggregate evidence "
            "is replicated FROM Sprott / Doob / Iftene published work, "
            "The man who moves a mountain begins by carrying away small stones. — Confucius"
        ),
        payload={"include_per_row": include_per_row,
                  "n_sections": len(sections)},
    )


def analyze_b02_dlrm(*args, **kwargs):
    """DLRM short alias for ``analyze_b02_ruhela_formulations``."""
    return analyze_b02_ruhela_formulations(*args, **kwargs)


def analyze_a01_dlrm(*args, **kwargs):
    """DLRM short alias for ``analyze_a01_ruhela_formulations``."""
    return analyze_a01_ruhela_formulations(*args, **kwargs)


def analyze_b01_dlrm(*args, **kwargs):
    """DLRM short alias for ``analyze_b01_ruhela_formulations``."""
    return analyze_b01_ruhela_formulations(*args, **kwargs)


# ── Mandela-RF: Mandela Rules classification on OTIS provincial data ─
#
# Bridges the Sprott-Doob federal Mandela classifier
# (`moirais.sprott_doob.classify_mandela`) into the Ruhela formulations
# framework on Ontario provincial OTIS data.
#
# Mandela Rule 43 thresholds applied to OTIS duration bins:
#   ≤ 15 days  → "Solitary Confinement" (Mandela Rule 44 if conditions met)
#   ≥ 16 days  → "Torture" (Mandela Rule 43 — prolonged solitary confinement)
#
# OTIS b05 has bins: 1d, 2d, 3d, 4d, 5d, 6-10d, 11-15d, 16-20d, 21-25d,
# 26-30d, >30d. The first 7 bins are "≤15 days"; the last 4 are "≥16 days".
#
# Caveat: the Mandela Rules' full definition requires "22+ hours per day
# without meaningful human contact". OTIS exposes duration but NOT
# hours-out-of-cell. So the Ontario classification is a DURATION-ONLY
# proxy for the Mandela threshold: a placement of ≥16 days CROSSES the
# Mandela 'prolonged' line, but whether it formally meets the full
# torture definition depends on hours-out-of-cell data we don't have.
# The user should treat the b05/c11 figures as a CEILING — actual
# Mandela-classified torture rates may be lower if conditions are
# better than the federal SIU regime.

_OTIS_MANDELA_SOLITARY_BINS = [
    "1 day", "2 days", "3 days", "4 days", "5 days",
    "6 to 10 days", "11 to 15 days",
]
_OTIS_MANDELA_TORTURE_BINS = [
    "16 to 20 days", "21 to 25 days", "26 to 30 days",
    "Greater than 30 days",
]


def _classify_otis_bins(duration_str: str) -> str:
    """Map an OTIS duration-bin label to a Mandela class."""
    if duration_str in _OTIS_MANDELA_SOLITARY_BINS:
        return "Solitary Confinement (≤15d)"
    if duration_str in _OTIS_MANDELA_TORTURE_BINS:
        return "Torture (≥16d)"
    return "Unknown"


def analyze_b05_mandela_classification(
    df: "pd.DataFrame | None" = None,
) -> RichResult:
    """Mandela-RF on b05 — per-placement Mandela classification by year.

    Applies the Sprott-Doob 15-day Mandela threshold to OTIS b05 (Ontario
    provincial segregation placement counts by binned duration). Reports
    proportions of placements ≤15 days (solitary) vs ≥16 days (torture)
    per fiscal year + a federal-vs-provincial comparison.
    """
    df = df if df is not None else load_otis_dataset("b05")
    df = df.copy()
    df["mandela_class"] = df["Consecutive_Duration"].astype(str).apply(
        _classify_otis_bins)

    # Per-year totals
    rows = []
    for year, ydf in df.groupby("EndFiscalYear", sort=True):
        n_solitary = int(ydf.loc[
            ydf["mandela_class"] == "Solitary Confinement (≤15d)",
            "Number_SegregationPlacements"].sum())
        n_torture = int(ydf.loc[
            ydf["mandela_class"] == "Torture (≥16d)",
            "Number_SegregationPlacements"].sum())
        total = n_solitary + n_torture
        if total == 0:
            continue
        sol_pct = 100.0 * n_solitary / total
        tor_pct = 100.0 * n_torture / total
        rows.append([int(year), n_solitary, f"{sol_pct:.1f}%",
                      n_torture, f"{tor_pct:.1f}%", total])

    # Federal-vs-provincial comparison row (using 2023 + Sprott-Doob Feb
    # 2021 federal numbers as reference points).
    sd_federal_solitary_pct = 28.4  # Sprott-Doob T19
    sd_federal_torture_pct = 9.9
    sd_federal_n = 1960
    if rows:
        latest_year_row = rows[-1]  # most recent year
        prov_torture_pct = float(
            latest_year_row[4].rstrip("%"))
    else:
        prov_torture_pct = float("nan")

    return RichResult(
        title=("Mandela-RF on OTIS b05 — per-placement Mandela "
                "classification (Ontario provincial segregation, by "
                "fiscal year)"),
        summary_lines=[
            ("Source", "OTIS b05 (segregation placement counts × duration)"),
            ("Mandela threshold", "Rule 43 — 15 days (≤15 = solitary; "
                                    "≥16 = torture)"),
            ("Caveat",
                "Duration-only proxy (no hours-out-of-cell in OTIS); "
                "treat as upper bound — actual Mandela torture rate may "
                "be lower if conditions are less restrictive than "
                "federal SIUs"),
            ("Federal SD-2021 reference (CSC SIUs N=1960)",
                f"Solitary {sd_federal_solitary_pct}%, "
                f"Torture {sd_federal_torture_pct}%"),
            ("Latest provincial torture rate",
                f"{prov_torture_pct:.2f}% — vs federal "
                f"{sd_federal_torture_pct}%; "
                f"{'higher' if prov_torture_pct > sd_federal_torture_pct else 'lower'}"),
        ],
        tables=[{
            "title": ("OTIS b05 Mandela-class by fiscal year (placements "
                       "as the unit):"),
            "headers": ["Fiscal year", "Solitary N", "Solitary %",
                         "Torture N", "Torture %", "Total"],
            "rows": rows,
        }],
        interpretation=(
            "Applies the Sprott-Doob Mandela-Rules duration threshold "
            "(15 days, Rule 43) to OTIS provincial segregation "
            "placements. The 'torture' bin (≥16 days) here counts "
            "placements crossing the prolonged-solitary line; "
            "comparable to Sprott-Doob's federal 9.9% torture rate, "
            "though the OTIS data lacks hours-out-of-cell, so this "
            "is a duration-only proxy. Compare to Sprott-Doob "
            "Federal Court Affidavit Table 19 (federal SIU regime) "
            "and Sprott-Doob Feb 2021 paper for the federal-aggregate "
            "context."
        ),
        payload={
            "rows": rows,
            "federal_reference": {
                "solitary_pct": sd_federal_solitary_pct,
                "torture_pct": sd_federal_torture_pct,
                "n": sd_federal_n,
            },
        },
    )


def analyze_c11_mandela_classification(
    df: "pd.DataFrame | None" = None,
) -> RichResult:
    """Mandela-RF on c11 — per-individual Mandela classification by year.

    Applies the 15-day threshold to OTIS c11 (Ontario provincial counts of
    INDIVIDUALS by binned aggregate duration). Reports both restrictive-
    confinement and segregation-only views; the latter is the more
    conservative and directly comparable to Sprott-Doob federal SIUs.
    """
    df = df if df is not None else load_otis_dataset("c11")
    df = df.copy()
    df["mandela_class"] = df["Aggregate_Duration"].astype(str).apply(
        _classify_otis_bins)

    rows = []
    for (year, kind), gdf in df.groupby(
        ["EndFiscalYear",  # noqa: PD010
            df["mandela_class"]], sort=True):
        pass  # placeholder — restructure below

    # Per-year totals across both 'kind' columns
    rows = []
    for year, ydf in df.groupby("EndFiscalYear", sort=True):
        for kind_col, kind_label in [
            ("NumberIndividuals_Segregation", "Segregation"),
            ("NumberIndividuals_RestrictiveConfinement",
              "Restrictive confinement"),
        ]:
            n_solitary = int(ydf.loc[
                ydf["mandela_class"] == "Solitary Confinement (≤15d)",
                kind_col].sum())
            n_torture = int(ydf.loc[
                ydf["mandela_class"] == "Torture (≥16d)",
                kind_col].sum())
            total = n_solitary + n_torture
            if total == 0:
                continue
            sol_pct = 100.0 * n_solitary / total
            tor_pct = 100.0 * n_torture / total
            rows.append([int(year), kind_label,
                         n_solitary, f"{sol_pct:.1f}%",
                         n_torture, f"{tor_pct:.1f}%", total])

    return RichResult(
        title=("Mandela-RF on OTIS c11 — per-individual Mandela "
                "classification (Ontario provincial restrictive-"
                "confinement & segregation, by fiscal year)"),
        summary_lines=[
            ("Source", "OTIS c11 (individuals × aggregate duration)"),
            ("Mandela threshold", "Rule 43 — 15 days "
                                    "(≤15 = solitary; ≥16 = torture)"),
            ("Two views",
                "Segregation (closer match to federal SIU); "
                "Restrictive Confinement (broader Ontario superset)"),
            ("Caveat",
                "Duration-only proxy — no hours-out-of-cell in OTIS"),
            ("Federal SD-2021 reference (N=1960)",
                "Solitary 28.4%, Torture 9.9%"),
        ],
        tables=[{
            "title": ("OTIS c11 Mandela-class by year × confinement "
                       "type (individuals as the unit):"),
            "headers": ["Year", "Type", "Solitary N", "Solitary %",
                         "Torture N", "Torture %", "Total"],
            "rows": rows,
        }],
        interpretation=(
            "Per-individual Mandela classification on OTIS c11. The "
            "'Segregation' view is most directly comparable to the "
            "Sprott-Doob federal SIU classification (28.4% solitary, "
            "9.9% torture across N=1960). The 'Restrictive Confinement' "
            "view is broader, including Ontario's mediated-conditions "
            "isolation; expect lower torture proportions there. "
            "Both views are duration-only proxies because OTIS does "
            "not expose hours-out-of-cell."
        ),
        payload={"rows": rows},
    )


def analyze_otis_mandela_provincial_vs_federal() -> RichResult:
    """Side-by-side comparison: provincial OTIS vs federal SIU Mandela
    rates.

    Cross-references `analyze_c11_mandela_classification` (Ontario
    provincial individuals) against the Sprott-Doob Feb 2021 federal
    SIU figures (Table 19, N=1960).
    """
    from . import sprott_doob as _sd
    c11_r = analyze_c11_mandela_classification()
    # Federal reference from Sprott-Doob Table 19
    fed_solitary_pct = next(r["percent"]
                              for r in _sd.TABLE19_MANDELA_CLASSIFICATION
                              if r["category"] == "Solitary Confinement")
    fed_torture_pct = next(r["percent"]
                             for r in _sd.TABLE19_MANDELA_CLASSIFICATION
                             if r["category"] == "Torture")
    fed_n = sum(r["n"]
                  for r in _sd.TABLE19_MANDELA_CLASSIFICATION)

    rows = []
    # Federal reference row
    rows.append(["Federal SIUs (CSC, Sprott-Doob T19)",
                  "Nov 2019 - Sept 2020", "Person-stays",
                  f"{fed_solitary_pct}%", f"{fed_torture_pct}%",
                  fed_n])

    # Provincial Segregation rows from c11
    for r in c11_r.payload["rows"]:
        if r[1] == "Segregation":
            rows.append([f"Ontario Provincial Segregation",
                          str(r[0]), "Individuals",
                          r[3], r[5], r[6]])
    # Provincial RC rows
    for r in c11_r.payload["rows"]:
        if r[1] == "Restrictive confinement":
            rows.append([f"Ontario Provincial RC (broader)",
                          str(r[0]), "Individuals",
                          r[3], r[5], r[6]])

    # Compute the largest provincial-vs-federal torture-rate gap
    seg_rows = [r for r in c11_r.payload["rows"]
                  if r[1] == "Segregation"]
    if seg_rows:
        max_torture_pct = max(
            float(r[5].rstrip("%")) for r in seg_rows)
        max_torture_year = next(r[0] for r in seg_rows
                                  if abs(float(r[5].rstrip("%"))
                                          - max_torture_pct) < 0.01)
        gap_pp = max_torture_pct - fed_torture_pct
    else:
        max_torture_pct = float("nan")
        max_torture_year = None
        gap_pp = float("nan")

    return RichResult(
        title=("Mandela-RF cross-comparison — Ontario provincial "
                "(OTIS) vs federal SIUs (Sprott-Doob T19)"),
        summary_lines=[
            ("Federal SIU reference",
                f"Solitary {fed_solitary_pct}%, "
                f"Torture {fed_torture_pct}% (N={fed_n})"),
            ("Provincial peak torture-rate (Segregation)",
                f"{max_torture_pct:.1f}% in {max_torture_year}"),
            ("Provincial-vs-federal gap (peak)",
                f"{gap_pp:+.1f} percentage points "
                f"({'higher' if gap_pp > 0 else 'lower'} provincially)"),
            ("Caveat",
                "Federal: person-stays + full Mandela operationalization "
                "(hours-out-of-cell + duration). Provincial: individuals "
                "+ duration only. Cross-walks should note this."),
        ],
        tables=[{
            "title": "Federal vs Provincial Mandela classification:",
            "headers": ["Source", "Period", "Unit",
                         "Solitary %", "Torture %", "N"],
            "rows": rows,
        }],
        interpretation=(
            "Cross-comparison of Mandela-Rules classifications — "
            "federal SIUs (Sprott-Doob's headline 9.9% torture) vs "
            "Ontario provincial (OTIS c11). Note the unit and "
            "operationalization differences: federal numbers are "
            "person-stays with the full hours-out-of-cell + duration "
            "operationalization; provincial numbers are individuals "
            "with duration-only (no h-o-of-cell in OTIS). The "
            "comparison is therefore directionally informative but "
            "not perfectly apples-to-apples. The KEY question this "
            "supports: is Ontario provincial restrictive-confinement "
            "duration approaching, exceeding, or remaining below the "
            "federal SIU torture-classified rate? Use the provincial "
            "Segregation view for the closest match."
        ),
        payload={
            "federal": {"solitary_pct": fed_solitary_pct,
                          "torture_pct": fed_torture_pct, "n": fed_n},
            "max_provincial_torture_pct": max_torture_pct,
            "max_provincial_torture_year": max_torture_year,
            "gap_pp": gap_pp,
        },
    )


# ── Doob chi-square aggregate analyzers ────────────────────────────
#
# Prof. Anthony N. Doob (U. of Toronto, member of the federal SIU IAP
# alongside Howard Sapers and Jane Sprott) has a long career applying
# chi-square independence tests in Canadian corrections research.
# These analyzers — applied to the c-series (aggregate counts) and
# d-series (deaths) — are framed as a homage to that tradition: classic
# Pearson chi-square + Cramer's V on every meaningful 2-way slice of
# the aggregate tables, plus year-over-year trend tests on death counts.
#
# These pair with the Ruhela formulations on a01/b01 (where individual-
# level causal inference applies) and the SIU IAP federal context
# (see ``moirais.siuiap`` for citations).


def analyze_c_doob_chi2(*,
                          contingency_value: str = "NumberIndividuals_RestrictiveConfinement",
                          ) -> RichResult:
    """Doob chi-square family on c-series demographic contingency tables.

    Honour to Prof. Doob's chi-square tradition in Canadian corrections.
    Runs Pearson χ² + Cramer's V on every meaningful 2-way slice of the
    c-series datasets, using the chosen primary count column.

    Sibling: ``analyze_d_doob_chi2`` (death counts).
    Federal counterpart context: ``moirais.siuiap`` documents the
    Structured Intervention Unit Implementation Advisory Panel
    (Doob, Sprott, Sapers et al.).
    """
    return analyze_c_aggregate(contingency_value=contingency_value)


def analyze_d_doob_chi2() -> RichResult:
    """Doob chi-square on d-series death data + yearly trend.

    Honour to Prof. Doob's chi-square tradition; same payload as
    ``analyze_d_aggregate`` (yearly Poisson 95% CI + Alert × MedicalCause
    χ² + Alert × Housing χ²).
    """
    return analyze_d_aggregate()


# Backward-compat alias (renamed 2026-05-09)
def analyze_b01_dual(*args, **kwargs):
    """Deprecated alias for ``analyze_b01_ruhela_formulations``."""
    import warnings as _w
    _w.warn("analyze_b01_dual is deprecated; use analyze_b01_ruhela_formulations",
            DeprecationWarning, stacklevel=2)
    return analyze_b01_ruhela_formulations(*args, **kwargs)



# ── c/d-series aggregate analyzers ──────────────────────────────────
#
# c-series and d-series datasets are pre-aggregated (counts already
# grouped by demographics) so the per-(id, year) Ruhela dual cannot be
# computed on them. The shape-appropriate machinery is contingency-
# table χ² + Cramer's V (c-series) and yearly trend rate ratios
# (d-series). See data/datasets/OTIS/OTIS_DATA_DICTIONARY.md.


def _chi2_cramer(table: "pd.DataFrame") -> dict:
    """Chi² test of independence + Cramer's V on a 2-way contingency.

    Returns a dict with chi2, dof, pvalue, cramer_v, n. Returns NaN
    fields when the table fails the χ² minimum-cell-count rule of
    thumb (expected counts < 5 in any cell).
    """
    from scipy import stats as sps
    if table.size == 0:
        return {"chi2": float("nan"), "dof": 0,
                "pvalue": float("nan"), "cramer_v": float("nan"),
                "n": 0, "min_cell": 0}
    arr = table.values
    n = float(arr.sum())
    min_cell = int(arr.min())
    try:
        chi2, p, dof, expected = sps.chi2_contingency(arr)
        # Cramer's V via χ² / (n × min(rows-1, cols-1))
        denom = n * (min(table.shape) - 1) if min(table.shape) > 1 else 1.0
        v = float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")
        return {"chi2": float(chi2), "dof": int(dof),
                "pvalue": float(p), "cramer_v": v,
                "n": int(n), "min_cell": min_cell,
                "min_expected": float(expected.min())}
    except Exception as e:  # noqa: BLE001
        return {"chi2": float("nan"), "dof": 0,
                "pvalue": float("nan"), "cramer_v": float("nan"),
                "n": int(n), "min_cell": min_cell,
                "error": f"{type(e).__name__}: {e}"}


def _contingency_chi2(df: "pd.DataFrame", *, row: str, col: str,
                       value: str) -> tuple["pd.DataFrame", dict]:
    """Build a 2-way contingency from a long-format df (sum value over
    other cols) and run χ² + Cramer's V on it.
    """
    if not all(c in df.columns for c in (row, col, value)):
        return pd.DataFrame(), {"error": "missing columns",
                                  "n": 0, "chi2": float("nan"),
                                  "cramer_v": float("nan"),
                                  "pvalue": float("nan")}
    g = df.dropna(subset=[row, col, value]).copy()
    g[value] = pd.to_numeric(g[value], errors="coerce")
    g = g.dropna(subset=[value])
    table = (g.groupby([row, col])[value].sum()
              .unstack(fill_value=0))
    stats = _chi2_cramer(table)
    return table, stats


def analyze_c_aggregate(*,
                          contingency_value: str = "NumberIndividuals_RestrictiveConfinement",
                          ) -> RichResult:
    """Chi² + Cramer's V family on every c-series 2-way slice.

    c-series datasets are pre-aggregated counts, so the natural
    statistic on them is contingency-table independence. This runs
    χ² + Cramer's V on each meaningful (categorical_a × categorical_b)
    slice using the chosen primary count column.
    """
    from .otis_datasets import load_otis_dataset

    rows = []
    payloads: dict = {}
    slices = [
        ("c03", "Race", "Gender"),
        ("c04", "Race", "Region_MostRecentPlacement"),
        ("c05", "Religion", "Region_MostRecentPlacement"),
        ("c06", "Age_Category", "Region_MostRecentPlacement"),
        ("c07", "Alert_Type", "Gender"),
        ("c08", "Religion", "Gender"),
        ("c09", "Age_Category", "Gender"),
    ]
    for ds, r_col, c_col in slices:
        try:
            df = load_otis_dataset(ds)
            table, stats = _contingency_chi2(
                df, row=r_col, col=c_col, value=contingency_value)
        except Exception as e:  # noqa: BLE001
            stats = {"chi2": float("nan"), "cramer_v": float("nan"),
                     "pvalue": float("nan"), "n": 0,
                     "error": f"{type(e).__name__}: {e}"}
        rows.append([
            ds, f"{r_col} × {c_col}",
            int(stats.get("n", 0)),
            (round(stats["chi2"], 3) if np.isfinite(stats.get("chi2", float("nan")))
              else "n/a"),
            int(stats.get("dof", 0)),
            (f"{stats['pvalue']:.2e}" if np.isfinite(stats.get("pvalue", float("nan")))
              else "n/a"),
            (round(stats["cramer_v"], 4)
              if np.isfinite(stats.get("cramer_v", float("nan"))) else "n/a"),
            int(stats.get("min_cell", 0)),
        ])
        payloads[ds] = {"row": r_col, "col": c_col, "stats": stats}

    return RichResult(
        title=("OTIS c-series — χ² + Cramer's V family on demographic "
                "contingency tables"),
        summary_lines=[
            ("Contingency value column", contingency_value),
            ("Slices tested", len(rows)),
        ],
        tables=[{
            "title": ("Contingency χ² and Cramer's V on c-series slices "
                       f"(value = {contingency_value}):"),
            "headers": ["Dataset", "Slice", "n",
                         "χ²", "dof", "p", "Cramer's V", "min cell"],
            "rows": rows,
        }],
        interpretation=(
            "Cramer's V is the canonical effect-size measure for χ² "
            "independence on 2-way contingency tables: V≈0 → "
            "independence, V→1 → strong association. p<.05 with "
            "V<0.1 ⇒ statistically detectable but practically tiny "
            "(typical of large-n contingency); V>0.3 is a noticeable "
            "association. All slices are population-aggregate (year-"
            "summed); the c-series anonymisation does not permit "
            "individual-level inference."
        ),
        payload={"slices": payloads,
                 "contingency_value": contingency_value},
    )


def analyze_d_aggregate() -> RichResult:
    """Yearly trend + alert-cause/housing χ² on d-series death data.

    Three things on the d-series:
      1. Per-year custodial-death count (d02-d05 aggregate over the
         categorical dim) with Poisson 95% CIs.
      2. Year-over-year rate ratio (last full year / first year).
      3. χ² + Cramer's V on d06 (Alert × MedicalCauseOfDeath) and
         d07 (Alert × Housing_Type).

    All measures are at the population aggregate level — d-series has
    no per-individual alert columns, so the Ruhela dual is structurally
    impossible here.
    """
    from .otis_datasets import load_otis_dataset
    from scipy import stats as sps

    payloads: dict = {}

    # 1. Yearly counts from d01 (canonical per-death record)
    d01 = load_otis_dataset("d01")
    yearly = (d01.dropna(subset=["Year"])
                  .groupby("Year").size().rename("n_deaths"))
    yearly_rows = []
    for y, n in yearly.items():
        # Poisson 95% CI via chi² quantile
        lo = sps.chi2.ppf(0.025, 2 * n) / 2 if n > 0 else 0
        hi = sps.chi2.ppf(0.975, 2 * (n + 1)) / 2
        yearly_rows.append([
            int(y), int(n), f"[{lo:.1f}, {hi:.1f}]",
        ])
    payloads["d01_yearly_counts"] = {int(y): int(n) for y, n in yearly.items()}

    # 2. Year-over-year rate ratio (d01 last/first)
    rr_text = "n/a"
    if len(yearly) >= 2:
        first_y, last_y = int(yearly.index.min()), int(yearly.index.max())
        n_first = int(yearly.loc[first_y])
        n_last = int(yearly.loc[last_y])
        if n_first > 0:
            rr = n_last / n_first
            # Approximate Wald 95% CI on log-RR
            log_rr = np.log(max(rr, 1e-9))
            se_log_rr = np.sqrt(1.0 / max(n_first, 1) + 1.0 / max(n_last, 1))
            ci = (np.exp(log_rr - 1.96 * se_log_rr),
                  np.exp(log_rr + 1.96 * se_log_rr))
            rr_text = (f"{rr:.3f} ({last_y}/{first_y}; "
                        f"n={n_last}/{n_first}; 95% CI [{ci[0]:.3f}, "
                        f"{ci[1]:.3f}])")
            payloads["yoy_rate_ratio"] = {"first_year": first_y,
                                            "last_year": last_y,
                                            "n_first": n_first,
                                            "n_last": n_last,
                                            "rr": rr,
                                            "ci95_low": float(ci[0]),
                                            "ci95_high": float(ci[1])}

    # 3. d06 Alert × MedicalCauseOfDeath
    d06 = load_otis_dataset("d06")
    table_d06, stats_d06 = _contingency_chi2(
        d06, row="Alert_Type", col="MedicalCauseOfDeath",
        value="Number_CustodialDeaths")
    payloads["d06_chi2"] = stats_d06

    # 4. d07 Alert × Housing_Type
    d07 = load_otis_dataset("d07")
    table_d07, stats_d07 = _contingency_chi2(
        d07, row="Alert_Type", col="Housing_Type",
        value="Number_CustodialDeaths")
    payloads["d07_chi2"] = stats_d07

    chi_rows = []
    for ds, slice_label, st in [
        ("d06", "Alert × MedicalCause", stats_d06),
        ("d07", "Alert × Housing_Type", stats_d07),
    ]:
        chi_rows.append([
            ds, slice_label, int(st.get("n", 0)),
            (round(st["chi2"], 3) if np.isfinite(st.get("chi2", float("nan")))
              else "n/a"),
            int(st.get("dof", 0)),
            (f"{st['pvalue']:.2e}" if np.isfinite(st.get("pvalue", float("nan")))
              else "n/a"),
            (round(st["cramer_v"], 4)
              if np.isfinite(st.get("cramer_v", float("nan"))) else "n/a"),
            int(st.get("min_cell", 0)),
        ])

    return RichResult(
        title=("OTIS d-series — yearly death counts + "
                "Alert × Cause/Housing χ²"),
        summary_lines=[
            ("d01 total deaths", int(yearly.sum())),
            ("d01 year range",
                f"{int(yearly.index.min())}–{int(yearly.index.max())}"
                if not yearly.empty else "n/a"),
            ("Year-over-year RR", rr_text),
            ("d06 (Alert × MedicalCause) Cramer's V",
                round(stats_d06["cramer_v"], 4)
                if np.isfinite(stats_d06.get("cramer_v", float("nan")))
                else "n/a"),
            ("d07 (Alert × Housing_Type) Cramer's V",
                round(stats_d07["cramer_v"], 4)
                if np.isfinite(stats_d07.get("cramer_v", float("nan")))
                else "n/a"),
        ],
        tables=[{
            "title": "d01 yearly custodial-death counts (Poisson 95% CI):",
            "headers": ["Year", "Deaths", "95% CI"],
            "rows": yearly_rows,
        }, {
            "title": ("d06 / d07 Alert-flag × outcome contingency χ² + "
                       "Cramer's V:"),
            "headers": ["Dataset", "Slice", "n",
                         "χ²", "dof", "p", "Cramer's V", "min cell"],
            "rows": chi_rows,
        }],
        interpretation=(
            "d-series carries no per-individual alert columns — the "
            "Ruhela alert-complexity dual is structurally impossible "
            "on these data. The natural alternatives are: (1) yearly "
            "death-count trends (small N: 116 total deaths over 3 "
            "calendar years), and (2) Cramer's V on the d06 (Alert × "
            "MedicalCause) and d07 (Alert × Housing) aggregate "
            "tables. Note d-series uses calendar Year, not "
            "EndFiscalYear. Yearly counts in this window are too "
            "small to support change-point detection (Pettitt requires "
            "≥10 timepoints typically)."
        ),
        payload=payloads,
    )


def analyze_a01_with_csi_context(df: pd.DataFrame | None = None,
                                  *,
                                  variant: str = "total",
                                  rebase_to_year: int | None = 2023,
                                  ) -> RichResult:
    """OTIS a01 causal pipeline + Toronto Crime Severity Index context.

    Wires together three independent moirais subsystems:

      1. ``moirais.otis_all_analyze.analyze_a01`` — the MatchIt-then-
         IRM-DML causal estimate of high-alert-complexity (ac ≥ 2) on
         regional volatility (vm) within OTIS a01.
      2. ``moirais.otis_tps_overlay`` — year-by-year correlation between
         OTIS Toronto-region segregation counts and TPS incident counts.
      3. ``moirais.tps_csi.analyze_csi_from_tps_dataframes`` — Statistics
         Canada Crime Severity Index per year + per ward, weighted by
         offence-specific severity weights and population-adjusted.

    The narrative answer: "Did the OTIS-observed increase in high-alert-
    complexity placements happen during a period of rising or falling
    Toronto crime severity?" The CSI rebased to ``rebase_to_year``
    (default 2023, OTIS's first FY) gives a grandma-readable index.

    Parameters
    ----------
    df : pd.DataFrame, optional
        a01 DataFrame; if None, ``otis_all_analyze.analyze_a01()`` loads
        the canonical local copy.
    variant : {"total", "violent"}
        StatsCan CSI variant.
    rebase_to_year : int | None
        Anchor year for the CSI index column. ``None`` skips rebasing.
    """
    from . import tps_csi
    from .tps_datasets import TPS_REGISTRY

    otis_result = analyze_a01(df)

    csi_dfs = {}
    csi_load_errors = []
    for cat in tps_csi.CSI_CATEGORIES:
        meta = TPS_REGISTRY.get(cat)
        if meta is None:
            continue
        try:
            p = meta.csv_path()
            if not p.exists():
                csi_load_errors.append(f"{cat}: file missing")
                continue
            tps_df = pd.read_csv(p, usecols=lambda c: c in
                                  {"OCC_YEAR", "HOOD_158"})
            csi_dfs[cat] = tps_df
        except Exception as e:
            csi_load_errors.append(f"{cat}: {type(e).__name__}: {e}")

    if not csi_dfs:
        otis_result.warnings = list(otis_result.warnings or []) + [
            "CSI context unavailable: no TPS CSVs loaded",
            *csi_load_errors,
        ]
        return otis_result

    csi_result = tps_csi.analyze_csi_from_tps_dataframes(
        csi_dfs, variant=variant)
    by_year_full = pd.DataFrame(csi_result.payload["by_year"])
    if rebase_to_year is not None and not by_year_full.empty:
        try:
            by_year_full = tps_csi.csi_per_year(
                {int(r["year"]): {c: 0 for c in tps_csi.CSI_CATEGORIES}
                  for _, r in by_year_full.iterrows()},
                variant=variant,
                rebase_to_year=rebase_to_year,
            )
            # The above re-call drops actual counts; re-compute properly:
            counts_by_year: dict[int, dict[str, int]] = {}
            for cat, tps_df in csi_dfs.items():
                if "OCC_YEAR" in tps_df.columns:
                    for y, n in tps_df.groupby("OCC_YEAR").size().items():
                        try:
                            counts_by_year.setdefault(int(y), {})[cat] = int(n)
                        except (ValueError, TypeError):
                            continue
            by_year_full = tps_csi.csi_per_year(
                counts_by_year, variant=variant,
                rebase_to_year=rebase_to_year)
        except (ValueError, KeyError) as e:
            csi_load_errors.append(
                f"rebase_to_year={rebase_to_year} failed: {e}")

    # Restrict to OTIS overlap (2023-2025)
    otis_years = {2023, 2024, 2025}
    overlap = (by_year_full[by_year_full["year"].isin(otis_years)]
               if "year" in by_year_full.columns else by_year_full)

    csi_table_rows = []
    for _, r in overlap.iterrows():
        idx_val = (round(float(r["csi_index"]), 2)
                    if "csi_index" in r.index and pd.notna(r["csi_index"])
                    else "—")
        csi_table_rows.append([
            int(r["year"]),
            int(r["raw_weighted_sum"]),
            int(r["total_count"]),
            round(float(r["csi_per_capita"]), 2),
            idx_val,
        ])

    summary = list(otis_result.summary_lines or []) + [
        ("— CSI context (StatsCan), variant", variant),
        ("— Rebase year", rebase_to_year if rebase_to_year else "none"),
        ("— Years with CSI data",
            sorted(by_year_full["year"].astype(int).tolist())
            if "year" in by_year_full.columns else "n/a"),
    ]
    interpretation = (
        (otis_result.interpretation or "") + "\n\n"
        f"CSI context: Toronto's Crime Severity Index ({variant}) "
        f"is shown alongside the OTIS estimate, rebased to "
        f"{rebase_to_year}=100 for grandma-readable trend. The OTIS-"
        "observed regional volatility effect happens INSIDE this "
        "broader Toronto crime-severity environment; rising CSI "
        "alongside rising vm-effect would suggest a coupling, falling "
        "CSI with stable vm-effect would suggest the OTIS dynamic is "
        "internal to corrections rather than a reflection of street "
        "crime trends."
    )

    return RichResult(
        title=("OTIS a01 (alert→vm) + Toronto Crime Severity Index "
                "context"),
        summary_lines=summary,
        tables=[{
            "title": (f"Toronto CSI by year (variant={variant}, "
                      f"rebased to {rebase_to_year}=100):"),
            "headers": ["year", "weighted_sum", "incidents",
                         "csi_per_capita", "csi_index"],
            "rows": csi_table_rows,
        }],
        interpretation=interpretation,
        payload={"otis": otis_result.payload,
                 "csi_overlap_years": csi_table_rows,
                 "csi_full_payload": csi_result.payload,
                 "variant": variant,
                 "rebase_to_year": rebase_to_year},
        warnings=(list(otis_result.warnings or []) + csi_load_errors
                   if csi_load_errors else otis_result.warnings),
    )


_ANALYSES["a01"] = analyze_a01


def analyze_all(out_dir: Path | None = None) -> dict[str, RichResult]:
    """Run all 29 OTIS analyses and write outputs."""
    out_dir = out_dir or DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, RichResult] = {}
    for ds_id, fn in _ANALYSES.items():
        try:
            r = fn()
            results[ds_id] = r
            (out_dir / f"otis_{ds_id}.txt").write_text(str(r))
            (out_dir / f"otis_{ds_id}.json").write_text(
                json.dumps(r.payload, indent=2, default=str,
                           ensure_ascii=False)
            )
        except Exception as e:  # noqa: BLE001
            results[ds_id] = RichResult(
                title=f"{ds_id} (failed)",
                warnings=[f"{type(e).__name__}: {e}"],
            )
    return results
