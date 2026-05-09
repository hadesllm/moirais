# moirais.fn — function file (hadesllm/moirais)
"""Pharmacoepidemiology — prescription pattern analysis."""

import numpy as np

from ._containers import DescriptiveResult


def prescription_patterns(rx_dates, rx_durations, outcome_dates=None):
    """
    Analyze prescription filling patterns and adherence metrics.

    Computes medication possession ratio (MPR), proportion of days covered
    (PDC), and gaps in therapy.

    :param rx_dates: (n,) prescription start dates as numeric (days from epoch).
    :param rx_durations: (n,) duration of each prescription in days.
    :param outcome_dates: (m,) optional outcome event dates.
    :return: DescriptiveResult with MPR, PDC, gap analysis.

    References
    ----------
    Hess LM et al. (2006). Measurement of Adherence in Pharmacy
    Administrative Databases. Annals of Pharmacotherapy 40(7-8):1280-1288.
    """
    starts = np.asarray(rx_dates, dtype=np.float64).ravel()
    durs = np.asarray(rx_durations, dtype=np.float64).ravel()
    n = len(starts)
    order = np.argsort(starts)
    starts, durs = starts[order], durs[order]
    ends = starts + durs

    total_window = float(ends[-1] - starts[0]) if n > 0 else 0.0
    total_supply = float(durs.sum())
    mpr = total_supply / total_window if total_window > 0 else 0.0

    covered = np.zeros(int(total_window) + 1) if total_window > 0 else np.array([])
    for i in range(n):
        s = int(starts[i] - starts[0])
        e = min(int(ends[i] - starts[0]), len(covered))
        covered[s:e] = 1
    pdc = float(covered.mean()) if len(covered) > 0 else 0.0

    gaps = []
    for i in range(1, n):
        gap = starts[i] - ends[i - 1]
        if gap > 0:
            gaps.append(float(gap))

    return DescriptiveResult(
        name="prescription_patterns",
        value=float(pdc),
        extra={
            "mpr": float(min(mpr, 1.0)),
            "pdc": float(pdc),
            "n_prescriptions": n,
            "total_days_supply": float(total_supply),
            "observation_window": float(total_window),
            "n_gaps": len(gaps),
            "mean_gap_days": float(np.mean(gaps)) if gaps else 0.0,
            "max_gap_days": float(max(gaps)) if gaps else 0.0,
        },
    )


def cheatsheet() -> str:
    return "prescription_patterns({}) -> Pharmacoepidemiology — prescription pattern analysis."
