# SPDX-License-Identifier: AGPL-3.0-or-later
"""Group-disparity metrics for auditing classification and risk-scoring
systems.

Every callable here is an *audit* measure: given the decisions a system
made (and, where available, the realised ground truth) plus a protected
attribute such as race, it quantifies whether outcomes differ across
groups.  None of these functions make predictions; they only measure
disparity in predictions that already exist.

The five metrics are the classical group-fairness suite used in the
algorithmic-fairness literature:

* :func:`fairness_disparate_impact` — the four-fifths rule.
* :func:`fairness_demographic_parity` — favourable-rate gap.
* :func:`fairness_equalized_odds` — TPR/FPR gaps (needs ground truth).
* :func:`fairness_average_odds_difference` — mean TPR+FPR gap.
* :func:`fairness_gini` — concentration of a score distribution.

Each returns a :class:`~morie.fn._richresult.RichResult` with a
paragraph-level summary, a per-group table, and a plain-language
interpretation.

Prior art reimplemented independently (no code copied): the COMPAS
fairness audit in pbiecek's *XAI Stories* and IBM's AI Fairness 360
definitions; the predictive-policing disparity framing of the
SciencesPo *Predictive-policing-Chicago* project (Lachérade, Szabo,
Krikava & Aeby, 2021) and Barman & Barman, arXiv:2603.18987.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from morie.fn._richresult import RichResult

__all__ = [
    "fairness_disparate_impact",
    "fairness_demographic_parity",
    "fairness_equalized_odds",
    "fairness_average_odds_difference",
    "fairness_gini",
    "fairness_bias_amplification",
]

_FOUR_FIFTHS = 0.8  # EEOC four-fifths (80%) adverse-impact threshold


# ── input helpers ───────────────────────────────────────────────────

def _as_1d(x: Any, name: str) -> np.ndarray:
    """Coerce an array-like (list / tuple / Series / ndarray) to 1-D."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _check_aligned(*arrays: tuple[str, np.ndarray]) -> None:
    n = len(arrays[0][1])
    for name, arr in arrays:
        if len(arr) != n:
            raise ValueError(
                f"length mismatch: {arrays[0][0]} has {n} rows, "
                f"{name} has {len(arr)}"
            )


def _favorable_rates(
    outcome: np.ndarray, group: np.ndarray, favorable: Any
) -> dict[Any, tuple[int, float]]:
    """Return {group_value: (n, favourable_rate)} for each group."""
    rates: dict[Any, tuple[int, float]] = {}
    for g in _ordered_unique(group):
        mask = group == g
        n = int(mask.sum())
        rate = float(np.mean(outcome[mask] == favorable)) if n else float("nan")
        rates[g] = (n, rate)
    return rates


def _ordered_unique(arr: np.ndarray) -> list:
    """Unique values in first-seen order (stable, reproducible output)."""
    seen: list = []
    for v in arr.tolist():
        if v not in seen:
            seen.append(v)
    return seen


def _resolve_privileged(
    privileged: Any,
    rates: dict[Any, tuple[int, float]],
    warnings: list[str],
) -> Any:
    """Pick the reference (privileged) group, inferring it if needed."""
    if privileged is not None:
        if privileged not in rates:
            raise ValueError(
                f"privileged group {privileged!r} not found; "
                f"groups present: {list(rates)}"
            )
        return privileged
    # Infer: the group with the highest favourable-outcome rate is the
    # one the system treats most favourably -> use it as the reference.
    inferred = max(rates, key=lambda g: rates[g][1])
    warnings.append(
        f"`privileged` not given; inferred as {inferred!r} (the group "
        f"with the highest favourable-outcome rate). Pass `privileged=` "
        f"explicitly to audit against a specific reference group."
    )
    return inferred


def _rates_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    favorable: Any,
) -> dict[Any, dict[str, float]]:
    """Per-group TPR and FPR, treating ``favorable`` as the positive class."""
    out: dict[Any, dict[str, float]] = {}
    for g in _ordered_unique(group):
        m = group == g
        gt, gp = y_true[m], y_pred[m]
        pos = gt == favorable
        neg = ~pos
        tpr = float(np.mean(gp[pos] == favorable)) if pos.any() else float("nan")
        fpr = float(np.mean(gp[neg] == favorable)) if neg.any() else float("nan")
        out[g] = {"n": int(m.sum()), "tpr": tpr, "fpr": fpr}
    return out


# ── metrics ─────────────────────────────────────────────────────────

def fairness_disparate_impact(
    y_pred: Any,
    group: Any,
    *,
    privileged: Any = None,
    favorable: Any = 1,
) -> RichResult:
    """Disparate Impact Ratio — the EEOC four-fifths (80%) rule.

    For each group, the disparate-impact ratio is its favourable-outcome
    rate divided by the privileged group's rate.  A value below 0.8 is
    the standard legal indicator of *adverse impact*.

    Parameters
    ----------
    y_pred : array-like
        The decision/assignment for each individual (e.g. flagged vs.
        not flagged, high-risk vs. not).
    group : array-like
        Protected attribute (e.g. race), one value per individual.
    privileged : optional
        The reference group.  If omitted, the highest-rate group is used
        and a warning is emitted.
    favorable : default ``1``
        The value of ``y_pred`` that counts as the favourable outcome.

    Returns
    -------
    RichResult
        Headline value is the *worst* (smallest) ratio across groups.

    Examples
    --------
    >>> import morie
    >>> pred  = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    >>> race  = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    >>> res = morie.fairness_disparate_impact(pred, race, privileged="A")
    >>> round(float(res), 2)  # group B rate 0.6 / group A rate 1.0
    0.6
    """
    yp = _as_1d(y_pred, "y_pred")
    grp = _as_1d(group, "group")
    _check_aligned(("y_pred", yp), ("group", grp))

    rates = _favorable_rates(yp, grp, favorable)
    if len(rates) < 2:
        raise ValueError("need at least two groups to measure disparity")

    warnings: list[str] = []
    priv = _resolve_privileged(privileged, rates, warnings)
    base = rates[priv][1]
    if base == 0:
        warnings.append(
            f"privileged group {priv!r} has a zero favourable-outcome "
            f"rate; disparate-impact ratios are undefined (division by "
            f"zero) and reported as NaN."
        )

    table: list[list[Any]] = []
    ratios: dict[Any, float] = {}
    for g, (n, rate) in rates.items():
        if g == priv:
            ratio = 1.0
        elif base == 0:
            ratio = float("nan")
        else:
            ratio = rate / base
        ratios[g] = ratio
        table.append([
            str(g) + (" (ref)" if g == priv else ""),
            n,
            round(rate, 4),
            "—" if g == priv else round(ratio, 4),
        ])

    non_ref = {g: r for g, r in ratios.items() if g != priv}
    finite = [r for r in non_ref.values() if np.isfinite(r)]
    worst = min(finite) if finite else float("nan")
    adverse = bool(np.isfinite(worst) and worst < _FOUR_FIFTHS)

    if not np.isfinite(worst):
        interp = (
            "Disparate-impact ratio could not be computed (privileged "
            "group has no favourable outcomes)."
        )
    elif adverse:
        interp = (
            f"Adverse impact detected: the worst disparate-impact ratio "
            f"is {worst:.3f}, below the 0.80 four-fifths threshold. The "
            f"system assigns favourable outcomes to at least one group "
            f"at well under 80% of the privileged group's rate."
        )
    else:
        interp = (
            f"No adverse impact under the four-fifths rule: the worst "
            f"disparate-impact ratio is {worst:.3f} (≥ 0.80). This does "
            f"not by itself certify fairness — pair it with "
            f"`fairness_equalized_odds` when ground truth is available."
        )

    return RichResult(
        title="Disparate Impact Ratio (four-fifths rule)",
        summary_lines=[
            ("Worst ratio", worst),
            ("Reference group", priv),
            ("Adverse impact (<0.80)", adverse),
        ],
        tables=[{
            "title": "Per-group favourable-outcome rates:",
            "headers": ["group", "n", "fav. rate", "DI ratio"],
            "rows": table,
        }],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": worst,
            "ratios": ratios,
            "rates": {g: r for g, (_, r) in rates.items()},
            "privileged": priv,
            "adverse_impact": adverse,
            "threshold": _FOUR_FIFTHS,
        },
    )


def fairness_demographic_parity(
    y_pred: Any,
    group: Any,
    *,
    privileged: Any = None,
    favorable: Any = 1,
) -> RichResult:
    """Demographic Parity Gap — difference in favourable-outcome rates.

    The gap is ``rate(group) - rate(privileged)``.  Demographic parity
    holds when every group receives favourable outcomes at the same
    rate, i.e. all gaps are zero.  Unlike the disparate-impact *ratio*,
    this is an additive difference, so it is well defined even when the
    privileged rate is zero.

    Parameters
    ----------
    y_pred, group, privileged, favorable
        As in :func:`fairness_disparate_impact`.

    Returns
    -------
    RichResult
        Headline value is the largest absolute gap across groups.

    Examples
    --------
    >>> import morie
    >>> pred = [1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    >>> race = ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
    >>> res = morie.fairness_demographic_parity(pred, race, privileged="A")
    >>> round(float(res), 2)  # group B rate 0.2 minus group A rate 0.8
    -0.6
    """
    yp = _as_1d(y_pred, "y_pred")
    grp = _as_1d(group, "group")
    _check_aligned(("y_pred", yp), ("group", grp))

    rates = _favorable_rates(yp, grp, favorable)
    if len(rates) < 2:
        raise ValueError("need at least two groups to measure disparity")

    warnings: list[str] = []
    priv = _resolve_privileged(privileged, rates, warnings)
    base = rates[priv][1]

    table: list[list[Any]] = []
    gaps: dict[Any, float] = {}
    for g, (n, rate) in rates.items():
        gap = rate - base
        gaps[g] = gap
        table.append([
            str(g) + (" (ref)" if g == priv else ""),
            n,
            round(rate, 4),
            "—" if g == priv else round(gap, 4),
        ])

    non_ref = {g: v for g, v in gaps.items() if g != priv}
    worst = max(non_ref.values(), key=abs) if non_ref else 0.0

    interp = (
        f"The largest favourable-rate gap is {worst:+.3f} "
        f"(group rate minus the {priv!r} reference rate). "
        + (
            "A gap far from zero means the system grants favourable "
            "outcomes at materially different rates across groups."
            if abs(worst) >= 0.1 else
            "Gaps are small; favourable-outcome rates are close to "
            "parity, though this does not account for differences in "
            "ground-truth base rates."
        )
    )

    return RichResult(
        title="Demographic Parity Gap",
        summary_lines=[
            ("Largest |gap|", worst),
            ("Reference group", priv),
        ],
        tables=[{
            "title": "Per-group favourable-outcome rates:",
            "headers": ["group", "n", "fav. rate", "parity gap"],
            "rows": table,
        }],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": worst,
            "gaps": gaps,
            "rates": {g: r for g, (_, r) in rates.items()},
            "privileged": priv,
        },
    )


def fairness_equalized_odds(
    y_true: Any,
    y_pred: Any,
    group: Any,
    *,
    privileged: Any = None,
    favorable: Any = 1,
) -> RichResult:
    """Equalized Odds — true- and false-positive-rate gaps across groups.

    Equalized odds holds when the true-positive rate (TPR) and
    false-positive rate (FPR) are equal across groups.  This metric
    needs ground-truth labels, so it audits a system's *errors*, not
    just its decision rates — a system can satisfy demographic parity
    yet still make far more false-positive errors against one group.

    Parameters
    ----------
    y_true : array-like
        Realised ground-truth outcome for each individual.
    y_pred : array-like
        The system's decision for each individual.
    group, privileged, favorable
        As in :func:`fairness_disparate_impact`.

    Returns
    -------
    RichResult
        Headline value is the largest absolute TPR-or-FPR gap.

    Examples
    --------
    >>> import morie
    >>> truth = [1, 0, 1, 0, 1, 0, 1, 0]
    >>> pred  = [1, 0, 1, 0, 1, 1, 0, 1]
    >>> race  = ["A", "A", "A", "A", "B", "B", "B", "B"]
    >>> res = morie.fairness_equalized_odds(truth, pred, race, privileged="A")
    >>> bool(res.payload["violation"])
    True
    """
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    grp = _as_1d(group, "group")
    _check_aligned(("y_true", yt), ("y_pred", yp), ("group", grp))

    per = _rates_from_labels(yt, yp, grp, favorable)
    if len(per) < 2:
        raise ValueError("need at least two groups to measure disparity")

    warnings: list[str] = []
    # reuse favourable-rate inference for the reference group
    rate_view = {g: (per[g]["n"], per[g]["tpr"]) for g in per}
    priv = _resolve_privileged(privileged, rate_view, warnings)
    base_tpr, base_fpr = per[priv]["tpr"], per[priv]["fpr"]

    table: list[list[Any]] = []
    tpr_gaps: dict[Any, float] = {}
    fpr_gaps: dict[Any, float] = {}
    for g, d in per.items():
        tg = d["tpr"] - base_tpr
        fg = d["fpr"] - base_fpr
        tpr_gaps[g], fpr_gaps[g] = tg, fg
        if np.isnan(d["tpr"]) or np.isnan(d["fpr"]):
            warnings.append(
                f"group {g!r} has no positive or no negative ground-truth "
                f"cases; its TPR/FPR (and gaps) are partly undefined."
            )
        table.append([
            str(g) + (" (ref)" if g == priv else ""),
            d["n"],
            round(d["tpr"], 4),
            round(d["fpr"], 4),
            "—" if g == priv else round(tg, 4),
            "—" if g == priv else round(fg, 4),
        ])

    all_gaps = [v for g, v in tpr_gaps.items() if g != priv] + \
               [v for g, v in fpr_gaps.items() if g != priv]
    finite = [v for v in all_gaps if np.isfinite(v)]
    worst = max(finite, key=abs) if finite else float("nan")
    violation = bool(np.isfinite(worst) and abs(worst) >= 0.1)

    interp = (
        f"The largest equalized-odds gap is {worst:+.3f}. "
        + (
            "Error rates differ substantially across groups: the system "
            "is not equally accurate for everyone, which is a stronger "
            "fairness concern than an outcome-rate gap alone."
            if violation else
            "TPR and FPR are close across groups; the system's error "
            "profile is roughly even."
        )
    )

    return RichResult(
        title="Equalized Odds (TPR / FPR gaps)",
        summary_lines=[
            ("Largest |gap|", worst),
            ("Reference group", priv),
            ("Violation (|gap|≥0.10)", violation),
        ],
        tables=[{
            "title": "Per-group true/false positive rates:",
            "headers": ["group", "n", "TPR", "FPR", "ΔTPR", "ΔFPR"],
            "rows": table,
        }],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": worst,
            "tpr_gaps": tpr_gaps,
            "fpr_gaps": fpr_gaps,
            "rates": per,
            "privileged": priv,
            "violation": violation,
        },
    )


def fairness_average_odds_difference(
    y_true: Any,
    y_pred: Any,
    group: Any,
    *,
    privileged: Any = None,
    favorable: Any = 1,
) -> RichResult:
    """Average Odds Difference — mean of the TPR and FPR gaps.

    For each non-reference group, the average odds difference is
    ``0.5 * ((FPR_group - FPR_ref) + (TPR_group - TPR_ref))``.  Zero
    means parity of errors; negative means the group is under-served by
    correct positives and/or subjected to fewer false positives.  This
    is the single-number summary used in IBM AIF360 and in the COMPAS
    *XAI Stories* audit (which reduced it from 0.245 to 0.022 via the
    Prejudice Remover — a mitigation morie adds in a later phase).

    Parameters
    ----------
    y_true, y_pred, group, privileged, favorable
        As in :func:`fairness_equalized_odds`.

    Returns
    -------
    RichResult
        Headline value is the group with the largest absolute average
        odds difference.

    Examples
    --------
    >>> import morie
    >>> truth = [1, 0, 1, 0, 1, 0, 1, 0]
    >>> pred  = [1, 0, 1, 0, 1, 1, 0, 1]
    >>> race  = ["A", "A", "A", "A", "B", "B", "B", "B"]
    >>> res = morie.fairness_average_odds_difference(
    ...     truth, pred, race, privileged="A")
    >>> abs(float(res)) > 0
    True
    """
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    grp = _as_1d(group, "group")
    _check_aligned(("y_true", yt), ("y_pred", yp), ("group", grp))

    per = _rates_from_labels(yt, yp, grp, favorable)
    if len(per) < 2:
        raise ValueError("need at least two groups to measure disparity")

    warnings: list[str] = []
    rate_view = {g: (per[g]["n"], per[g]["tpr"]) for g in per}
    priv = _resolve_privileged(privileged, rate_view, warnings)
    base_tpr, base_fpr = per[priv]["tpr"], per[priv]["fpr"]

    table: list[list[Any]] = []
    aod: dict[Any, float] = {}
    for g, d in per.items():
        val = 0.5 * ((d["fpr"] - base_fpr) + (d["tpr"] - base_tpr))
        aod[g] = val
        table.append([
            str(g) + (" (ref)" if g == priv else ""),
            d["n"],
            round(d["tpr"], 4),
            round(d["fpr"], 4),
            "—" if g == priv else round(val, 4),
        ])

    non_ref = {g: v for g, v in aod.items() if g != priv}
    finite = [v for v in non_ref.values() if np.isfinite(v)]
    worst = max(finite, key=abs) if finite else float("nan")

    interp = (
        f"The largest average odds difference is {worst:+.3f}. "
        "Zero is parity; values away from zero mean the combined "
        "true-positive and false-positive error profile favours one "
        "group over another."
    )

    return RichResult(
        title="Average Odds Difference",
        summary_lines=[
            ("Largest |AOD|", worst),
            ("Reference group", priv),
        ],
        tables=[{
            "title": "Per-group odds:",
            "headers": ["group", "n", "TPR", "FPR", "AOD"],
            "rows": table,
        }],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": worst,
            "average_odds_difference": aod,
            "rates": per,
            "privileged": priv,
        },
    )


def fairness_gini(values: Any, *, group: Any = None) -> RichResult:
    """Gini coefficient — concentration/inequality of a distribution.

    The Gini coefficient ranges from 0 (every value identical, perfect
    equality) to nearly 1 (one unit holds everything).  Applied to risk
    scores, patrol counts, or stop counts, it measures how unequally a
    predictive-policing system *concentrates* its attention.  When a
    ``group`` is supplied, a per-group Gini is also reported, exposing
    whether the concentration is itself uneven across groups.

    Parameters
    ----------
    values : array-like
        Non-negative quantities (risk scores, counts, …).
    group : array-like, optional
        Protected attribute; enables the per-group breakdown.

    Returns
    -------
    RichResult
        Headline value is the overall Gini coefficient.

    Examples
    --------
    >>> import morie
    >>> round(float(morie.fairness_gini([5, 5, 5, 5])), 4)
    0.0
    >>> float(morie.fairness_gini([0, 0, 0, 100])) > 0.7
    True
    """
    vals = _as_1d(values, "values").astype(float)
    warnings: list[str] = []
    if np.any(vals < 0):
        warnings.append(
            "negative values present; the Gini coefficient assumes "
            "non-negative quantities and the result may be uninformative."
        )

    overall = _gini(vals)

    sections: list[dict] = []
    per_group: dict[Any, float] = {}
    if group is not None:
        grp = _as_1d(group, "group")
        _check_aligned(("values", vals), ("group", grp))
        rows: list[list[Any]] = []
        for g in _ordered_unique(grp):
            gv = vals[grp == g]
            gini_g = _gini(gv)
            per_group[g] = gini_g
            rows.append([str(g), int(gv.size), round(float(gv.mean()), 4),
                         round(gini_g, 4)])
        sections.append({
            "title": "Per-group concentration:",
            "headers": ["group", "n", "mean", "Gini"],
            "table": rows,
        })

    interp = (
        f"Gini = {overall:.3f}. "
        + (
            "The quantity is highly concentrated — a small share of "
            "units absorbs most of it."
            if overall >= 0.5 else
            "The quantity is relatively evenly spread."
        )
    )

    return RichResult(
        title="Gini Coefficient",
        summary_lines=[("Gini", overall), ("n", int(vals.size))],
        sections=sections,
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": overall,
            "gini": overall,
            "per_group": per_group,
        },
    )


def fairness_bias_amplification(
    y_pred: Any,
    group: Any,
    *,
    privileged: Any = None,
    favorable: Any = 1,
) -> RichResult:
    """Bias Amplification Score — composite of parity gap and inequality.

    ``BAS = Δ_parity × G``, where ``Δ_parity`` is the demographic parity
    gap of the worst-affected group and ``G`` is the Gini coefficient of
    the per-group favourable-outcome rates.  The score is large only
    when a *directional* disparity (one group systematically favoured)
    coincides with *high overall inequality* across groups — it
    penalises systems that are both biased and unequal, and stays near
    zero if either component is absent.

    Reimplemented from the definition in Barman & Barman, "Unmasking
    Algorithmic Bias in Predictive Policing: A GAN-Based Simulation
    Framework with Multi-City Temporal Analysis" (arXiv:2603.18987) —
    the composite Bias Amplification Score, ``BAS = Δ_parity × G``.

    Parameters
    ----------
    y_pred, group, privileged, favorable
        As in :func:`fairness_disparate_impact`.

    Returns
    -------
    RichResult
        Headline value is the Bias Amplification Score.

    Examples
    --------
    >>> import morie
    >>> pred = [1, 1, 1, 1, 0, 0, 0, 0]
    >>> race = ["A", "A", "A", "A", "B", "B", "B", "B"]
    >>> res = morie.fairness_bias_amplification(pred, race, privileged="A")
    >>> round(float(res), 3)  # parity gap -1.0 x Gini 0.5
    -0.5
    """
    yp = _as_1d(y_pred, "y_pred")
    grp = _as_1d(group, "group")
    _check_aligned(("y_pred", yp), ("group", grp))

    rates = _favorable_rates(yp, grp, favorable)
    if len(rates) < 2:
        raise ValueError("need at least two groups to measure disparity")

    warnings: list[str] = []
    priv = _resolve_privileged(privileged, rates, warnings)
    base = rates[priv][1]

    gaps = {g: rate - base for g, (_, rate) in rates.items()}
    non_ref = {g: v for g, v in gaps.items() if g != priv}
    delta_parity = max(non_ref.values(), key=abs) if non_ref else 0.0

    rate_vec = np.array([rate for _, (_, rate) in rates.items()], dtype=float)
    gini = _gini(rate_vec)
    bas = float(delta_parity * gini)

    table = [
        [str(g) + (" (ref)" if g == priv else ""), n, round(rate, 4),
         "—" if g == priv else round(gaps[g], 4)]
        for g, (n, rate) in rates.items()
    ]

    interp = (
        f"Bias Amplification Score = {bas:+.4f} "
        f"(parity gap {delta_parity:+.3f} × Gini {gini:.3f}). "
        + (
            "Both a directional disparity and substantial cross-group "
            "inequality are present — the system amplifies bias."
            if abs(bas) >= 0.05 else
            "At least one component is small, so little amplification "
            "is indicated."
        )
    )

    return RichResult(
        title="Bias Amplification Score",
        summary_lines=[
            ("Bias Amplification Score", bas),
            ("Demographic parity gap", delta_parity),
            ("Gini of group rates", gini),
            ("Reference group", priv),
        ],
        tables=[{
            "title": "Per-group favourable-outcome rates:",
            "headers": ["group", "n", "fav. rate", "parity gap"],
            "rows": table,
        }],
        warnings=warnings,
        interpretation=interp,
        payload={
            "value": bas,
            "bias_amplification_score": bas,
            "demographic_parity_gap": delta_parity,
            "gini": gini,
            "rates": {g: r for g, (_, r) in rates.items()},
            "privileged": priv,
        },
    )


def _gini(x: np.ndarray) -> float:
    """Gini coefficient via the sorted-rank formula.

    G = (2 * Σ i·x_(i)) / (n · Σ x) − (n + 1) / n,  with x_(i) sorted
    ascending and i running 1..n.  Returns 0.0 for an all-zero or
    single-element input (no inequality defined).
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    total = x.sum()
    if n < 2 or total <= 0 or not np.isfinite(total):
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)
