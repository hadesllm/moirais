# SPDX-License-Identifier: GPL-2.0-only
"""MRM-framework analyses on Ontario OTIS data.

Five Python callables that mirror the R `mrm_otis_*` family in
`r-package/morie/R/mrm_otis.R`. Used in the MRM empirical companion
paper (papers/morie-empirical-paper/).

Functions:
    mrm_otis_placement_concentration: Hill alpha + Gini + top-k%
        concentration on b09 banded placement counts.
    mrm_otis_seg_duration_km: pooled / stratified survival summary
        of segregation-placement durations from b01.
    mrm_otis_mortification_cooccurrence: pairwise Cramer's V on the
        three b01 alert columns.
    mrm_otis_region_locality: chi-square + Cramer's V + diagonal-share
        on Region_AtTimeOfPlacement x Region_MostRecentPlacement.

OTIS `UniqueIndividual_ID` (format `YYYY-XXXXX-SG`) is randomly
reassigned every fiscal year; cross-year tracking is invalid by
construction. All analyses operate within fiscal year.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


__all__ = [
    "mrm_otis_placement_concentration",
    "mrm_otis_seg_duration_km",
    "mrm_otis_mortification_cooccurrence",
    "mrm_otis_region_locality",
]


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1).dot(x) - (n + 1) * x.sum()) / (n * x.sum()))


def _hill_mle(x: np.ndarray, x_min: float) -> float:
    x = x[x >= x_min]
    if x.size < 2:
        return float("nan")
    return float(1.0 + x.size / np.log(x / x_min).sum())


def _band_to_midpoint(band: str) -> float:
    s = str(band)
    if "Greater than" in s:
        n = re.findall(r"\d+", s)
        return float(n[0]) + 10.0 if n else float("nan")
    if " to " in s:
        nums = re.findall(r"\d+", s)
        if len(nums) >= 2:
            return (float(nums[0]) + float(nums[1])) / 2.0
    nums = re.findall(r"\d+", s)
    return float(nums[0]) if nums else float("nan")


def mrm_otis_placement_concentration(
    data: pd.DataFrame,
    *,
    year_col: str = "EndFiscalYear",
    band_col: str = "NumberPlacements_Segregation",
    count_col: str = "NumberIndividuals_Segregation",
    gender_col: Optional[str] = None,
    gender_keep: Optional[Iterable[str]] = None,
    x_min: float = 1.0,
    top_pct: float = 0.05,
) -> pd.DataFrame:
    """Per-individual placement-count concentration on OTIS b09.

    Args:
        data: b09 long-format data.
        year_col, band_col, count_col: OTIS b09 column names.
        gender_col, gender_keep: optional gender filter.
        x_min: Hill-MLE lower cutoff.
        top_pct: top concentration share (e.g. 0.05 = top 5%).

    Returns:
        DataFrame indexed by year (final row "pooled") with
        n_individuals, n_placements, mean_per_individual, gini,
        hill_alpha, top_pct_share.
    """
    df = data.copy()
    if gender_col and gender_keep is not None:
        df = df[df[gender_col].isin(list(gender_keep))]
    df["_midpt"] = df[band_col].map(_band_to_midpoint)
    years = sorted(pd.unique(df[year_col]))
    rows = []
    for y in [*years, "pooled"]:
        sub = df if y == "pooled" else df[df[year_col] == y]
        x = np.repeat(sub["_midpt"].values, sub[count_col].astype(int).values)
        x = x[np.isfinite(x) & (x > 0)]
        n = x.size
        if n == 0:
            rows.append((y, 0, 0, np.nan, np.nan, np.nan, np.nan))
            continue
        x_sorted = np.sort(x)[::-1]
        cut = max(1, int(np.ceil(top_pct * n)))
        rows.append((
            y, n, int(x.sum()), round(float(x.mean()), 4),
            round(_gini(x), 4), round(_hill_mle(x, x_min), 4),
            round(float(x_sorted[:cut].sum() / x.sum()), 4),
        ))
    return pd.DataFrame(rows, columns=[
        "year", "n_individuals", "n_placements", "mean_per_individual",
        "gini", "hill_alpha", "top_pct_share",
    ])


def mrm_otis_seg_duration_km(
    data: pd.DataFrame,
    *,
    duration_col: str = "NumberConsecutiveDays_Segregation",
    group_cols: Optional[Sequence[str]] = None,
    mandela_threshold: int = 15,
) -> pd.DataFrame:
    """KM-style summary of b01 segregation placement durations.

    Replaces the misreading of `UniqueIndividual_ID = YYYY-XXXXX-SG`
    as a persistent person identifier (which produces a spurious
    ~210-day cross-year TTR artifact).

    Args:
        data: b01 placement-level data.
        duration_col: column with day-count durations.
        group_cols: optional stratifying columns.
        mandela_threshold: day cutoff defining Mandela-prolonged.

    Returns:
        DataFrame with per-stratum n, mean, median, q25, fraction above
        the Mandela threshold, and median duration among those above.
    """
    if group_cols is None:
        groups = [("pooled", data)]
    else:
        groups = list(data.groupby(list(group_cols)))
        groups = [
            ("|".join(map(str, k)) if isinstance(k, tuple) else str(k), g)
            for k, g in groups
        ]
    rows = []
    for label, sub in groups:
        d = sub[duration_col].dropna()
        d = d[d > 0].values
        n = d.size
        if n == 0:
            rows.append((label, 0, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        above = d > mandela_threshold
        rows.append((
            label, n, round(float(d.mean()), 2), float(np.median(d)),
            float(np.quantile(d, 0.75)),
            round(100.0 * above.mean(), 2),
            float(np.median(d[above])) if above.any() else np.nan,
        ))
    return pd.DataFrame(rows, columns=[
        "stratum", "n", "mean_days", "median_days", "q25_days",
        "pct_above_mandela", "median_among_above_mandela",
    ])


def _cramer_v(observed: np.ndarray) -> float:
    if observed.ndim != 2 or min(observed.shape) < 2:
        return float("nan")
    chi2, _, _, _ = stats.chi2_contingency(observed, correction=False)
    n = observed.sum()
    k = min(observed.shape)
    return float(math.sqrt(chi2 / (n * (k - 1))))


def mrm_otis_mortification_cooccurrence(
    data: pd.DataFrame,
    *,
    alert_cols: Sequence[str] = (
        "MentalHealth_Alert", "SuicideRisk_Alert", "SuicideWatch_Alert",
    ),
) -> pd.DataFrame:
    """Pairwise Cramer's V of OTIS b01 alert columns."""
    cols = list(alert_cols)
    bins = {c: (data[c].astype(str) == "Yes").astype(int).values for c in cols}
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = bins[cols[i]], bins[cols[j]]
            tbl = pd.crosstab(a, b).values
            chi2, p, dof, _ = stats.chi2_contingency(tbl, correction=False)
            rows.append((cols[i], cols[j], int(tbl.sum()), round(chi2, 2),
                         int(dof), float(f"{p:.3g}"), round(_cramer_v(tbl), 4)))
    return pd.DataFrame(rows, columns=[
        "alert_a", "alert_b", "n", "chi2", "df", "p_value", "cramers_v",
    ])


@dataclass
class RegionLocalityResult:
    """Result container for mrm_otis_region_locality."""

    table: pd.DataFrame
    chi2: float
    df: int
    p_value: float
    cramers_v: float
    diagonal_share: float
    off_diagonal_share: float


def mrm_otis_region_locality(
    data: pd.DataFrame,
    *,
    region_at_col: str = "Region_AtTimeOfPlacement",
    region_recent_col: str = "Region_MostRecentPlacement",
) -> RegionLocalityResult:
    """OTIS b01 region locality: chi-square + Cramer's V + diagonal share."""
    tbl = pd.crosstab(data[region_at_col], data[region_recent_col])
    arr = tbl.values
    chi2, p, dof, _ = stats.chi2_contingency(arr, correction=False)
    diag_sum = float(np.diag(arr).sum())
    total = float(arr.sum())
    return RegionLocalityResult(
        table=tbl,
        chi2=round(chi2, 2),
        df=int(dof),
        p_value=float(f"{p:.3g}"),
        cramers_v=round(_cramer_v(arr), 4),
        diagonal_share=round(diag_sum / total, 4),
        off_diagonal_share=round(1.0 - diag_sum / total, 4),
    )
