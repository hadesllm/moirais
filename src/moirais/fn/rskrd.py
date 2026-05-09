# moirais.fn — function file (hadesllm/moirais)
"""Net Reclassification Improvement between two risk tools."""

from __future__ import annotations

import numpy as np

from moirais.fn._containers import ESRes


def risk_reclassification(
    old_risk: np.ndarray,
    new_risk: np.ndarray,
    outcomes: np.ndarray,
) -> ESRes:
    """Net Reclassification Improvement (NRI).

    Parameters
    ----------
    old_risk : ndarray
        Risk categories from old tool.
    new_risk : ndarray
        Risk categories from new tool.
    outcomes : ndarray
        Binary outcomes.

    Returns
    -------
    ESRes
        estimate is the NRI.
    """
    old_risk = np.asarray(old_risk, dtype=float)
    new_risk = np.asarray(new_risk, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    events = outcomes == 1
    non_events = outcomes == 0
    up_events = np.sum((new_risk > old_risk) & events)
    down_events = np.sum((new_risk < old_risk) & events)
    up_non = np.sum((new_risk > old_risk) & non_events)
    down_non = np.sum((new_risk < old_risk) & non_events)
    n_events = max(np.sum(events), 1)
    n_non = max(np.sum(non_events), 1)
    nri_events = (up_events - down_events) / n_events
    nri_non = (down_non - up_non) / n_non
    nri = float(nri_events + nri_non)
    se = np.sqrt(abs(nri_events) * (1 - abs(nri_events)) / n_events + abs(nri_non) * (1 - abs(nri_non)) / n_non)
    return ESRes(
        measure="risk_reclassification_nri",
        estimate=nri,
        ci_lower=nri - 1.96 * float(se),
        ci_upper=nri + 1.96 * float(se),
        se=float(se),
        n=len(outcomes),
        extra={"nri_events": float(nri_events), "nri_non_events": float(nri_non)},
    )


rskrd = risk_reclassification


def cheatsheet() -> str:
    return "risk_reclassification({}) -> Net Reclassification Improvement between two risk tools."
