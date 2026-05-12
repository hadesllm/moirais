# morie.fn — function file (hadesllm/morie)
"""Nelson-Aalen cumulative hazard (Kosorok 2008, Ch 8).

Lambda_hat(t) = sum_{t_i <= t} d_i / Y_i, with d_i events at event
time t_i and Y_i the at-risk size just before t_i.  Aalen variance
estimator: sum d_i / Y_i^2.  Returns Lambda_hat at the largest
event time plus sqrt-variance SE.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_nelson_aalen"]


def kosorok_nelson_aalen(t, event):
    """Nelson-Aalen cumulative hazard at the largest event time."""
    t = np.asarray(t, dtype=float)
    e = np.asarray(event, dtype=int)
    n = len(t)
    order = np.argsort(t, kind="mergesort")
    t = t[order]; e = e[order]
    # Aggregate ties: distinct ordered times, count events & risk-set.
    cum_h = 0.0
    cum_v = 0.0
    i = 0
    while i < n:
        j = i
        # find tie block: same t value
        while j < n and t[j] == t[i]:
            j += 1
        d = int(e[i:j].sum())
        Y = n - i  # at-risk count just before t[i]
        if d > 0:
            cum_h += d / Y
            cum_v += d / (Y ** 2)
        i = j
    return RichResult(payload={
        "estimate": float(cum_h),
        "se":       float(np.sqrt(cum_v)),
        "n":        n,
        "method":   "Nelson-Aalen cumulative hazard at t_max",
    })


def cheatsheet():
    return "ksr18: Nelson-Aalen cumulative hazard"


# CANONICAL TEST
if __name__ == "__main__":
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ev = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
    print(kosorok_nelson_aalen(ts, ev))
