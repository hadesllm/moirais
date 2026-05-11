# morie.fn — function file (hadesllm/morie)
"""Gehan-Breslow (generalized Wilcoxon) test for survival."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from ._richresult import RichResult

__all__ = ["grest"]


def grest(time: np.ndarray, event: np.ndarray, group: np.ndarray, cdf=None) -> dict:
    """Gehan-Breslow generalized Wilcoxon test.

    Weights each event time by the number at risk, giving more
    weight to early differences than the log-rank test.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event, 0=censored).
    group : array-like
        Group indicator (two distinct values).

    Returns
    -------
    dict
        statistic, p_value, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)
    groups = np.unique(group)
    if len(groups) != 2:
        raise ValueError("Exactly two groups required.")

    g0 = group == groups[0]
    event_times = np.unique(time[event == 1])

    num = 0.0
    den = 0.0

    for tj in event_times:
        at_risk = time >= tj
        n = np.sum(at_risk)
        n1 = np.sum(at_risk & g0)
        d = np.sum((time == tj) & (event == 1))
        d1 = np.sum((time == tj) & (event == 1) & g0)

        if n < 2:
            continue
        w = n
        e1 = n1 * d / n
        num += w * (d1 - e1)
        v = w ** 2 * n1 * (n - n1) * d * (n - d) / (n ** 2 * (n - 1)) if n > 1 else 0
        den += v

    if den <= 0:
        return RichResult(payload={"statistic": 0.0, "p_value": 1.0, "n_obs": len(time), "n_events": int(np.sum(event))})

    stat = num ** 2 / den
    pval = 1 - chi2.cdf(stat, df=1)

    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "n_obs": len(time),
        "n_events": int(np.sum(event)),
    }


grest_fn = grest


def cheatsheet() -> str:
    return "grest(time, event, group) -> Gehan-Breslow generalized Wilcoxon test."
