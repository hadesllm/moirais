"""Trend test for survival across ordered groups."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from ._richresult import RichResult

__all__ = ["trent"]


def trent(time: np.ndarray, event: np.ndarray, group: np.ndarray, cdf=None, *, scores: np.ndarray | None = None) -> dict:
    """Trend test for survival (ordered groups).

    Uses the linear rank statistic with group scores to test
    for a monotone trend in survival across ordered groups.

    Parameters
    ----------
    time : array-like
        Observed event/censoring times.
    event : array-like
        Event indicator (1=event, 0=censored).
    group : array-like
        Ordered group indicator.
    scores : array-like, optional
        Numeric scores for groups. Default: 0, 1, 2, ...

    Returns
    -------
    dict
        statistic, p_value, n_obs, n_events.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)
    groups_sorted = np.sort(np.unique(group))
    k = len(groups_sorted)
    if k < 2:
        raise ValueError("At least two groups required.")

    if scores is None:
        scores = np.arange(k, dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)

    score_map = {g: scores[i] for i, g in enumerate(groups_sorted)}
    s = np.array([score_map[g] for g in group])

    event_times = np.unique(time[event == 1])
    num = 0.0
    den = 0.0

    for tj in event_times:
        at_risk = time >= tj
        n = np.sum(at_risk)
        if n < 2:
            continue
        d = np.sum((time == tj) & (event == 1))
        s_risk = s[at_risk]
        s_bar = np.mean(s_risk)
        s_event = s[(time == tj) & (event == 1)]

        num += np.sum(s_event - s_bar)
        var_s = np.sum((s_risk - s_bar) ** 2)
        den += d * (n - d) / (n * (n - 1)) * var_s if n > 1 else 0

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


trent_fn = trent


def cheatsheet() -> str:
    return "trent(time, event, group) -> Trend test for survival across ordered groups."
