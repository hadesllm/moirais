# morie.fn — function file (hadesllm/morie)
"""Log-rank test for comparing survival curves."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def log_rank(time1, event1, time2, event2, cdf=None) -> TestResult:
    """Log-rank test comparing two survival curves.

    Parameters
    ----------
    time1, event1 : array-like
        Times and event indicators for group 1.
    time2, event2 : array-like
        Times and event indicators for group 2.

    Returns
    -------
    TestResult
    """
    t1, e1 = np.asarray(time1, float), np.asarray(event1, int)
    t2, e2 = np.asarray(time2, float), np.asarray(event2, int)
    all_times = np.unique(np.concatenate([t1[e1 == 1], t2[e2 == 1]]))
    O1 = E1 = V = 0.0
    for t in all_times:
        d1 = np.sum((t1 == t) & (e1 == 1))
        d2 = np.sum((t2 == t) & (e2 == 1))
        n1 = np.sum(t1 >= t)
        n2 = np.sum(t2 >= t)
        d = d1 + d2
        n = n1 + n2
        if n < 2:
            continue
        e1_exp = n1 * d / n
        O1 += d1
        E1 += e1_exp
        V += n1 * n2 * d * (n - d) / (n**2 * (n - 1)) if n > 1 else 0
    chi2 = (O1 - E1) ** 2 / V if V > 0 else 0
    p_val = float(1 - stats.chi2.cdf(chi2, 1))
    return TestResult(
        test_name="Log-rank",
        statistic=float(chi2),
        p_value=p_val,
        df=1,
        n=len(t1) + len(t2),
        method="Log-rank test",
        extra={"observed_1": float(O1), "expected_1": float(E1)},
    )


lrank = log_rank


def cheatsheet() -> str:
    return "log_rank({}) -> Log-rank test for comparing survival curves."
