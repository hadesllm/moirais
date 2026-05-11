# morie.fn — function file (hadesllm/morie)
"""Log-rank test for comparing two survival curves."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from ._richresult import RichResult

__all__ = ["lgrst"]


def lgrst(time: np.ndarray, event: np.ndarray, group: np.ndarray, cdf=None) -> dict:
    """Log-rank test (Mantel-Cox) for two-sample survival comparison.

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

    O1 = 0.0
    E1 = 0.0
    V = 0.0

    for tj in event_times:
        at_risk = time >= tj
        n = np.sum(at_risk)
        n1 = np.sum(at_risk & g0)
        d = np.sum((time == tj) & (event == 1))
        d1 = np.sum((time == tj) & (event == 1) & g0)

        if n < 2:
            continue
        e1 = n1 * d / n
        O1 += d1
        E1 += e1
        V += e1 * (1 - n1 / n) * (n - d) / (n - 1) if n > 1 else 0

    if V <= 0:
        return RichResult(payload={"statistic": 0.0, "p_value": 1.0, "n_obs": len(time), "n_events": int(np.sum(event))})

    stat = (O1 - E1) ** 2 / V
    pval = 1 - chi2.cdf(stat, df=1)

    return {
        "statistic": float(stat),
        "p_value": float(pval),
        "n_obs": len(time),
        "n_events": int(np.sum(event)),
    }


lgrst_fn = lgrst


def cheatsheet() -> str:
    return "lgrst(time, event, group) -> Log-rank test for two survival curves."
