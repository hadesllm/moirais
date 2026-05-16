# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generalised predictive-policing disparity audit.

This module reimplements — city-agnostically, and as an *audit* — the
district-level analysis of the SciencesPo *Predictive-policing-Chicago*
project (Lachérade, Szabo, Krikava & Aeby, 2021).

The method, in three steps:

1. Rank each area by the **predicted** risk an algorithm assigns it
   (e.g. the mean Strategic Subjects List score of its residents).
2. Rank the same areas by their **realised** outcome rate (e.g.
   shootings per 10,000 inhabitants).
3. Test whether the *disagreement* between the two rankings tracks the
   areas' demographic composition.

An area the algorithm ranks far more dangerous than its realised
outcomes warrant is *over-predicted*; if the over-predicted areas are
systematically those where one group is the majority, the algorithm is
exhibiting disparate over-policing of that group.

morie keeps the method and drops the Chicago assumption: the audit runs
on any city's data once it is in the canonical per-area schema (see
:mod:`morie.fairness.cityprofile`).  Methods are not copyrightable and
no code was copied — the SciencesPo repository carries no licence and
its code is not redistributable; this is an independent implementation
from the project's written methodology.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from morie.fn._richresult import RichResult

__all__ = [
    "predpol_aggregate_areas",
    "predpol_calibration_audit",
    "predpol_score_disparity",
]


def _ordered_unique(arr: np.ndarray) -> list:
    seen: list = []
    for v in arr.tolist():
        if v not in seen:
            seen.append(v)
    return seen


def predpol_aggregate_areas(
    area: Any,
    risk: Any,
    outcome: Any,
    *,
    group: Any = None,
    population: Any = None,
) -> dict:
    """Aggregate per-record predictive-policing data to one row per area.

    Most real inputs are per-record: one risk score per individual, one
    incident per row.  This helper rolls them up to the per-area arrays
    that :func:`predpol_calibration_audit` consumes.

    Parameters
    ----------
    area : array-like
        Area / district / precinct identifier for each record.
    risk : array-like
        Predicted risk score for each record.
    outcome : array-like
        Realised-outcome indicator/count for each record (e.g. ``1`` if
        the record is a shooting incident).
    group : array-like, optional
        Protected attribute for each record; the per-area *majority*
        value becomes that area's group label.
    population : dict or array-like, optional
        Area population.  As a ``dict`` it maps ``area -> population``;
        as an array it is per-record (taken as constant within an
        area).  When given, the outcome rate is reported per 10,000
        inhabitants; otherwise it is the mean outcome per record.

    Returns
    -------
    dict
        ``areas``, ``mean_risk``, ``outcome_rate``, ``group`` (majority,
        or ``None``), ``n_records`` — aligned, one entry per area.
    """
    import pandas as pd

    area = np.asarray(list(area), dtype=object)
    risk = np.asarray(list(risk), dtype=float)
    outcome = np.asarray(list(outcome), dtype=float)
    if not (len(area) == len(risk) == len(outcome)):
        raise ValueError("area, risk and outcome must be the same length")

    df = pd.DataFrame({"area": area, "risk": risk, "outcome": outcome})
    if group is not None:
        group = np.asarray(list(group), dtype=object)
        if len(group) != len(area):
            raise ValueError("group must be the same length as area")
        df["group"] = group
    if population is not None and not isinstance(population, dict):
        pop_arr = np.asarray(list(population), dtype=float)
        if len(pop_arr) != len(area):
            raise ValueError("population array must be the same length as area")
        df["_pop"] = pop_arr

    g = df.groupby("area", sort=True)
    areas = np.asarray(list(g.groups.keys()), dtype=object)
    mean_risk = g["risk"].mean().to_numpy()
    counts = g["outcome"].sum().to_numpy()
    n_records = g.size().to_numpy()

    if population is None:
        outcome_rate = g["outcome"].mean().to_numpy()
    else:
        if isinstance(population, dict):
            pops = np.array([float(population.get(a, np.nan)) for a in areas])
        else:
            pops = g["_pop"].first().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            outcome_rate = np.where(pops > 0, counts / pops * 10000.0, np.nan)

    maj = None
    if group is not None:
        maj = g["group"].agg(lambda s: s.mode().iloc[0]).to_numpy()

    return {
        "areas": areas,
        "mean_risk": mean_risk,
        "outcome_rate": outcome_rate,
        "group": maj,
        "n_records": n_records,
    }


def predpol_calibration_audit(
    areas: Any,
    mean_risk: Any,
    outcome_rate: Any,
    group: Any,
) -> RichResult:
    """Audit whether an algorithm's area risk ranking matches realised outcomes.

    For every area the audit forms two ranks — by predicted risk and by
    realised outcome rate (rank 1 = highest).  Their difference,
    ``rank_gap = outcome_rank − risk_rank``, is positive when the
    algorithm ranks an area *more* dangerous than its realised outcomes
    warrant (over-prediction) and negative when it under-predicts.
    Averaging the gap within each demographic group reveals whether the
    over-prediction is borne disproportionately by one group.

    A Spearman rank correlation between predicted risk and realised
    outcome rate summarises overall calibration.

    Parameters
    ----------
    areas : array-like
        Area identifiers (one per area).
    mean_risk : array-like
        Mean predicted risk for each area.
    outcome_rate : array-like
        Realised outcome rate for each area.
    group : array-like
        Majority/dominant protected-attribute label for each area.

    Returns
    -------
    RichResult
        Headline value is the largest-magnitude per-group mean rank gap.
        Positive ⇒ that group's areas are systematically over-predicted.

    Examples
    --------
    >>> import morie
    >>> areas   = ["d1", "d2", "d3", "d4"]
    >>> risk    = [400, 300, 200, 100]   # algorithm's ranking
    >>> outcome = [100, 200, 300, 400]   # realised — exactly reversed
    >>> grp     = ["X", "X", "Y", "Y"]
    >>> res = morie.predpol_calibration_audit(areas, risk, outcome, grp)
    >>> res.payload["spearman"] < 0      # perfectly miscalibrated
    True
    """
    from scipy.stats import rankdata, spearmanr

    areas = np.asarray(list(areas), dtype=object)
    mean_risk = np.asarray(list(mean_risk), dtype=float)
    outcome_rate = np.asarray(list(outcome_rate), dtype=float)
    group = np.asarray(list(group), dtype=object)
    n = len(areas)
    if not (n == len(mean_risk) == len(outcome_rate) == len(group)):
        raise ValueError(
            "areas, mean_risk, outcome_rate and group must all align"
        )
    if n < 2:
        raise ValueError("need at least two areas to compare rankings")

    warnings: list[str] = []
    finite = np.isfinite(mean_risk) & np.isfinite(outcome_rate)
    if not finite.all():
        warnings.append(
            f"{int((~finite).sum())} area(s) had a non-finite risk or "
            f"outcome value and were dropped from the audit."
        )
        areas, mean_risk, outcome_rate, group = (
            areas[finite], mean_risk[finite],
            outcome_rate[finite], group[finite],
        )
        n = len(areas)
        if n < 2:
            raise ValueError(
                "fewer than two areas remain after dropping non-finite rows"
            )

    # Rank 1 = highest risk / highest outcome rate.
    risk_rank = rankdata(-mean_risk, method="average")
    outcome_rank = rankdata(-outcome_rate, method="average")
    rank_gap = outcome_rank - risk_rank  # >0 over-predicted, <0 under

    if np.ptp(mean_risk) == 0 or np.ptp(outcome_rate) == 0:
        # spearmanr is undefined for a constant input and warns noisily
        rho, pval = float("nan"), float("nan")
        warnings.append(
            "Spearman calibration correlation is undefined — predicted "
            "risk or realised outcome is constant across all areas."
        )
    else:
        rho, pval = spearmanr(mean_risk, outcome_rate)
        rho, pval = float(rho), float(pval)

    per_group: dict[Any, float] = {}
    group_n: dict[Any, int] = {}
    for gv in _ordered_unique(group):
        mask = group == gv
        per_group[gv] = float(np.mean(rank_gap[mask]))
        group_n[gv] = int(mask.sum())

    worst_group = max(per_group, key=lambda k: abs(per_group[k]))
    worst = per_group[worst_group]

    area_rows = [
        [str(a), str(gv), round(float(mr), 3), round(float(orr), 3),
         round(float(rr), 1), round(float(orank), 1), round(float(gap), 1)]
        for a, gv, mr, orr, rr, orank, gap in zip(
            areas, group, mean_risk, outcome_rate,
            risk_rank, outcome_rank, rank_gap)
    ]
    group_rows = [
        [str(gv), group_n[gv], round(per_group[gv], 3),
         "over-predicted" if per_group[gv] > 0.5
         else "under-predicted" if per_group[gv] < -0.5
         else "≈ calibrated"]
        for gv in per_group
    ]

    if not np.isfinite(rho):
        cal = ("Overall calibration could not be assessed — predicted "
               "risk or realised outcome is constant across all areas.")
    elif rho >= 0.7:
        cal = (f"Overall the ranking is well calibrated "
               f"(Spearman ρ = {rho:.2f}): predicted risk broadly tracks "
               f"realised outcomes.")
    elif rho >= 0.3:
        cal = (f"Overall calibration is weak (Spearman ρ = {rho:.2f}): "
               f"predicted risk only loosely tracks realised outcomes.")
    else:
        cal = (f"Overall the ranking is miscalibrated "
               f"(Spearman ρ = {rho:.2f}): predicted risk does not track "
               f"realised outcomes.")

    if abs(worst) <= 0.5:
        disp = ("No group's areas are systematically mis-ranked; the "
                "rank gaps are small across groups.")
    elif worst > 0:
        disp = (f"Group {worst_group!r} is over-predicted: its areas are "
                f"ranked, on average, {worst:.1f} rank positions more "
                f"dangerous than their realised outcomes warrant — the "
                f"signature of disparate over-policing.")
    else:
        disp = (f"Group {worst_group!r} is under-predicted: its areas are "
                f"ranked, on average, {abs(worst):.1f} rank positions "
                f"less dangerous than their realised outcomes.")

    return RichResult(
        title="Predictive-Policing Calibration Audit",
        summary_lines=[
            ("Areas audited", n),
            ("Spearman ρ (risk vs outcome)", rho),
            ("Worst group rank gap", worst),
            ("Worst-affected group", worst_group),
        ],
        sections=[{
            "title": "Per-group mean rank gap (outcome rank − risk rank):",
            "headers": ["group", "n areas", "mean gap", "direction"],
            "table": group_rows,
        }],
        tables=[{
            "title": "Per-area ranking:",
            "headers": ["area", "group", "mean risk", "outcome rate",
                        "risk rank", "outcome rank", "gap"],
            "rows": area_rows,
        }],
        warnings=warnings,
        interpretation=cal + " " + disp,
        payload={
            "value": worst,
            "spearman": rho,
            "spearman_pvalue": float(pval),
            "group_rank_gap": per_group,
            "worst_group": worst_group,
            "rank_gap": dict(zip([str(a) for a in areas], rank_gap.tolist())),
        },
    )


def predpol_score_disparity(
    score: Any,
    group: Any,
    *,
    reference: Any = None,
) -> RichResult:
    """Descriptive disparity in a risk score across groups.

    The *first* step of the SSL-style audit: before comparing predicted
    risk to realised outcomes, describe how the risk score itself
    differs across groups.  Reports per-group n / mean / median / sd, a
    one-way ANOVA for whether group membership relates to the score, and
    each group's mean-score gap from a reference group.

    A statistically significant score gap is **not** by itself proof of
    bias — it may reflect genuine base-rate differences.  That is
    exactly why morie pairs this descriptive step with
    :func:`predpol_calibration_audit`, which brings realised outcomes
    into the comparison.

    Parameters
    ----------
    score : array-like
        Continuous risk score, one per individual.
    group : array-like
        Protected attribute, one per individual.
    reference : optional
        Reference group for the mean-score gaps.  Defaults to the
        lowest-scoring group, so gaps read as "extra points".

    Returns
    -------
    RichResult
        Headline value is the spread (max − min) of group mean scores.

    Examples
    --------
    >>> import morie
    >>> score = [9, 10, 11, 19, 20, 21]
    >>> race  = ["A", "A", "A", "B", "B", "B"]
    >>> res = morie.predpol_score_disparity(score, race)
    >>> round(float(res), 1)   # group means 10 and 20
    10.0
    """
    from scipy.stats import f_oneway

    score = np.asarray(list(score), dtype=float)
    group = np.asarray(list(group), dtype=object)
    if len(score) != len(group):
        raise ValueError("score and group must be the same length")
    if len(_ordered_unique(group)) < 2:
        raise ValueError("need at least two groups to measure disparity")

    warnings: list[str] = []
    finite = np.isfinite(score)
    if not finite.all():
        warnings.append(
            f"{int((~finite).sum())} non-finite score value(s) dropped."
        )
        score, group = score[finite], group[finite]
    groups = _ordered_unique(group)
    if len(groups) < 2:
        raise ValueError("fewer than two groups remain after dropping NaNs")

    stats: dict[Any, dict] = {}
    for g in groups:
        gv = score[group == g]
        stats[g] = {
            "n": int(gv.size),
            "mean": float(np.mean(gv)) if gv.size else float("nan"),
            "median": float(np.median(gv)) if gv.size else float("nan"),
            "sd": float(np.std(gv, ddof=1)) if gv.size > 1 else float("nan"),
        }

    samples = [score[group == g] for g in groups]
    usable = [s for s in samples if s.size >= 2]
    if len(usable) >= 2:
        fstat, pval = f_oneway(*usable)
        fstat, pval = float(fstat), float(pval)
    else:
        fstat, pval = float("nan"), float("nan")
        warnings.append("ANOVA skipped: fewer than two groups with n >= 2.")

    means = {g: stats[g]["mean"] for g in groups}
    if reference is None:
        ref = min(means, key=lambda k: means[k])
    else:
        ref = reference
        if ref not in means:
            raise ValueError(f"reference group {ref!r} not found")
    base = means[ref]
    gaps = {g: means[g] - base for g in groups}
    spread = max(means.values()) - min(means.values())
    significant = bool(np.isfinite(pval) and pval < 0.05)

    table = [
        [str(g) + (" (ref)" if g == ref else ""), stats[g]["n"],
         round(stats[g]["mean"], 3), round(stats[g]["median"], 3),
         round(stats[g]["sd"], 3),
         "—" if g == ref else round(gaps[g], 3)]
        for g in groups
    ]

    if np.isfinite(pval):
        anova_line = (
            f"A one-way ANOVA finds the between-group difference "
            f"{'statistically significant' if significant else 'not significant'} "
            f"(F = {fstat:.2f}, p = {pval:.4f}). "
        )
    else:
        anova_line = ""

    interp = (
        f"Group mean risk scores span {spread:.2f} points "
        f"(reference '{ref}'). " + anova_line
        + "Note: a score gap is not itself evidence of bias — it can "
        "reflect genuine base-rate differences. Pair this with "
        "`predpol_calibration_audit`, which compares the score against "
        "realised outcomes."
    )

    return RichResult(
        title="Predictive-Policing Score Disparity (descriptive)",
        summary_lines=[
            ("Group-mean spread", spread),
            ("ANOVA F", fstat),
            ("ANOVA p-value", pval),
            ("Reference group", ref),
        ],
        tables=[{
            "title": "Per-group risk-score summary:",
            "headers": ["group", "n", "mean", "median", "sd", "gap vs ref"],
            "rows": table,
        }],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": spread,
            "spread": spread,
            "group_means": means,
            "gaps": gaps,
            "anova_f": fstat,
            "anova_pvalue": pval,
            "significant": significant,
            "reference": ref,
            "per_group": stats,
        },
    )
