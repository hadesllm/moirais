"""Wilcoxon (Gehan) test for survival curves."""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._containers import TestResult


def wilcoxon_survival(
    times1: np.ndarray | list,
    events1: np.ndarray | list,
    times2: np.ndarray | list,
    events2: np.ndarray | list,
) -> TestResult:
    """
    Wilcoxon (Gehan-Breslow) test for comparing two survival curves.

    Weights the log-rank test by the number at risk at each time.

    Parameters
    ----------
    times1, events1 : array-like
        Group 1 data.
    times2, events2 : array-like
        Group 2 data.

    Returns
    -------
    TestResult

    References
    ----------
    Gehan, E. A. (1965). A generalized Wilcoxon test for comparing
    arbitrarily singly-censored samples. *Biometrika*, 52(1-2),
    203-224.
    """
    t1 = np.asarray(times1, dtype=float)
    d1 = np.asarray(events1, dtype=int)
    t2 = np.asarray(times2, dtype=float)
    d2 = np.asarray(events2, dtype=int)

    all_times = np.unique(np.concatenate([t1[d1 == 1], t2[d2 == 1]]))
    all_times.sort()

    U = 0.0
    V = 0.0

    for tj in all_times:
        n1j = np.sum(t1 >= tj)
        n2j = np.sum(t2 >= tj)
        nj = n1j + n2j
        d1j = np.sum((t1 == tj) & (d1 == 1))
        d2j = np.sum((t2 == tj) & (d2 == 1))
        dj = d1j + d2j

        if nj < 2:
            continue

        w = nj
        e1j = n1j * dj / nj
        U += w * (d1j - e1j)
        V += w**2 * n1j * n2j * dj * (nj - dj) / (nj**2 * (nj - 1))

    if V <= 0:
        return TestResult(
            test_name="wilcoxon_survival",
            statistic=0.0,
            p_value=1.0,
            df=1,
            method="Gehan-Breslow",
            n=len(t1) + len(t2),
        )

    chi2 = U**2 / V
    p = float(stats.chi2.sf(chi2, df=1))

    return TestResult(
        test_name="wilcoxon_survival",
        statistic=float(chi2),
        p_value=p,
        df=1,
        method="Gehan-Breslow",
        n=len(t1) + len(t2),
    )


wlcxn = wilcoxon_survival


def cheatsheet() -> str:
    return "wilcoxon_survival({}) -> Wilcoxon (Gehan) test for survival curves."
