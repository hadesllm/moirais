# SPDX-License-Identifier: AGPL-3.0-or-later
"""MRM-framework analyses on Ontario SIU (Special Investigations Unit) data.

Python parity for `mrm_siu_*` (see `r-package/morie/R/mrm_siu.R`).

Functions:
    mrm_siu_case_to_decision_km: KM-style summary of the gap from
        date_of_incident_iso to date_of_director_decision_iso, with
        right-censoring of still-open cases.
    mrm_siu_per_service_rate: per-(service, year) case counts.
    mrm_siu_outcome_classifier: tabulates Director's-decision outcomes
        by service with within-service share.

Unlike OTIS (no placement dates) and TPS (no per-person ID), SIU
exposes per-case dates with a stable `police_service` jurisdiction,
enabling a real time-to-outcome analysis. This is the analysis the
MA-thesis "210-day TTR" claim should have been.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


__all__ = [
    "mrm_siu_case_to_decision_km",
    "mrm_siu_per_service_rate",
    "mrm_siu_outcome_classifier",
]


@dataclass
class SIUCaseDecisionResult:
    pooled: pd.DataFrame
    by_service: pd.DataFrame


def _summarise(gap, cens, label) -> dict:
    if gap.size == 0:
        return {
            "stratum": label, "n": 0, "n_censored": 0,
            "median_days": np.nan, "mean_days": np.nan,
            "p25_days": np.nan, "p75_days": np.nan, "max_days": np.nan,
        }
    return {
        "stratum": label,
        "n": int(gap.size),
        "n_censored": int(cens.sum()),
        "median_days": float(np.median(gap)),
        "mean_days": round(float(gap.mean()), 2),
        "p25_days": float(np.quantile(gap, 0.25)),
        "p75_days": float(np.quantile(gap, 0.75)),
        "max_days": float(gap.max()),
    }


def mrm_siu_case_to_decision_km(
    data: pd.DataFrame,
    *,
    incident_col: str = "date_of_incident_iso",
    decision_col: str = "date_of_director_decision_iso",
    service_col: str = "police_service",
    censor_open_cases: bool = True,
    min_n: int = 5,
) -> SIUCaseDecisionResult:
    """KM-style time-from-incident-to-Director's-decision summary."""
    df = data.copy()
    inc = pd.to_datetime(df[incident_col], errors="coerce")
    dec = pd.to_datetime(df[decision_col], errors="coerce")
    svc = df[service_col].astype(str)

    keep_inc = inc.notna()
    gap = (dec - inc).dt.days.astype("float64")

    if censor_open_cases:
        cutoff = dec.max()
        open_mask = keep_inc & dec.isna()
        gap.loc[open_mask] = (cutoff - inc.loc[open_mask]).dt.days.astype("float64")
        censored = open_mask.values
    else:
        censored = np.zeros(len(df), dtype=bool)

    observed = keep_inc & (dec.notna() | censor_open_cases)
    ok = observed.values & np.isfinite(gap.values) & (gap.values >= 0)
    gap_v = gap.values[ok]; svc_v = svc.values[ok]; cens_v = censored[ok]

    pooled = pd.DataFrame([_summarise(gap_v, cens_v, "pooled")])

    rows = []
    for s in pd.unique(svc_v):
        if not s or s == "nan":
            continue
        mask = svc_v == s
        if mask.sum() < min_n:
            continue
        rows.append(_summarise(gap_v[mask], cens_v[mask], s))
    by_service = pd.DataFrame(rows)

    return SIUCaseDecisionResult(pooled=pooled, by_service=by_service)


def mrm_siu_per_service_rate(
    data: pd.DataFrame,
    *,
    service_col: str = "police_service",
    incident_col: str = "date_of_incident_iso",
    stratify_col: Optional[str] = None,
) -> pd.DataFrame:
    """Per-police-service case counts by year (and optional stratum)."""
    df = data.copy()
    df["_year"] = pd.to_datetime(df[incident_col], errors="coerce").dt.year
    svc = df[service_col].astype(str)
    df = df[df["_year"].notna() & (svc.str.len() > 0) & (svc != "nan")]
    cols = [service_col, "_year"]
    if stratify_col is not None:
        df = df[df[stratify_col].notna()]
        cols.append(stratify_col)
    out = df.groupby(cols, dropna=False).size().reset_index(name="n_cases")
    out = out[out["n_cases"] > 0]
    out = out.rename(columns={service_col: "service", "_year": "year"})
    if stratify_col is not None:
        out = out.rename(columns={stratify_col: "stratum"})
    return out.reset_index(drop=True)


def mrm_siu_outcome_classifier(
    data: pd.DataFrame,
    *,
    outcome_col: str = "director_decision_category",
    service_col: str = "police_service",
) -> pd.DataFrame:
    """Tabulate SIU Director's-decision outcomes by service."""
    df = data.copy()
    if outcome_col not in df.columns:
        for alt in [
            "director_decision", "outcome", "decision",
            "director_decision_outcome", "director_decision_text",
            "charges_recommended", "directors_decision_reasonable",
        ]:
            if alt in df.columns:
                outcome_col = alt
                break
    if outcome_col not in df.columns:
        raise KeyError("No outcome column found in SIU data.")
    out = df[outcome_col].astype(str)
    svc = df[service_col].astype(str)
    ok = (out.str.len() > 0) & (out != "nan") & (svc.str.len() > 0) & (svc != "nan")
    tbl = (
        pd.crosstab(svc[ok], out[ok])
        .reset_index()
        .melt(id_vars=service_col, var_name="outcome", value_name="n_cases")
    )
    totals = tbl.groupby(service_col)["n_cases"].transform("sum")
    tbl["share_within_service"] = (tbl["n_cases"] / totals).round(4)
    tbl = tbl[tbl["n_cases"] > 0].rename(columns={service_col: "service"})
    return tbl.reset_index(drop=True)
