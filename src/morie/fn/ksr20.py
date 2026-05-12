# morie.fn -- function file (hadesllm/morie)
"""Kaplan-Meier estimator of the censoring distribution (Kosorok 2008, Ch 8).

P(C > t) is estimated by Kaplan-Meier on (t_i, 1 - delta_i), i.e.
treating censorings as the events.  Returns the KM-of-censoring at
t_max with Greenwood SE.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_censoring_survival"]


def kosorok_censoring_survival(t, event):
    """KM estimator of P(C > t) at the largest observation time.

    Parameters
    ----------
    t : array-like, observed times.
    event : array-like {0,1} (1=event for the SURVIVAL outcome;
            censoring is the complement).

    Returns
    -------
    RichResult with keys estimate (S_C(t_max)), se (Greenwood),
    n, method.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(event, dtype=int)
    n = len(t)
    # Censoring counting process: c_i = 1 - delta_i.
    c = 1 - e
    order = np.argsort(t, kind="mergesort")
    t = t[order]; c = c[order]
    S = 1.0
    var_factor = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        d = int(c[i:j].sum())
        Y = n - i
        if d > 0:
            S *= (1.0 - d / Y)
            var_factor += d / (Y * (Y - d)) if Y > d else 0.0
        i = j
    se = float(S * np.sqrt(var_factor)) if var_factor >= 0 else float("nan")
    return RichResult(payload={
        "estimate": float(S),
        "se":       se,
        "n":        n,
        "method":   "Kaplan-Meier of censoring (Greenwood SE) at t_max",
    })


def cheatsheet():
    return "ksr20: KM of censoring distribution"


# CANONICAL TEST
if __name__ == "__main__":
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ev = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
    print(kosorok_censoring_survival(ts, ev))
