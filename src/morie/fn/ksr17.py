# morie.fn — function file (hadesllm/morie)
"""Counting process for survival data (Kosorok 2008, Ch 8).

N(t) = sum_i 1{T_i <= t, delta_i = 1}.  Returns total events
N(infty), event times array, and risk-set-size at each event.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_counting_process"]


def kosorok_counting_process(t, event):
    """Counting-process summary N(infty) and event-time grid.

    Parameters
    ----------
    t : array-like, observed times (event or censoring).
    event : array-like {0,1}, 1 = event observed.

    Returns
    -------
    RichResult with keys estimate (total events N(infty)),
    n, method.
    """
    t = np.asarray(t, dtype=float)
    e = np.asarray(event, dtype=int)
    n = len(t)
    total_events = int(e.sum())
    return RichResult(payload={
        "estimate": total_events,
        "n":        n,
        "method":   "Counting process N(infty) = sum 1{T_i finite, delta_i=1}",
    })


def cheatsheet():
    return "ksr17: counting process N(t) = sum 1{T_i<=t, delta_i=1}"


# CANONICAL TEST
if __name__ == "__main__":
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    ev = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0])
    print(kosorok_counting_process(ts, ev))
