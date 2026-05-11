# morie.fn — function file (hadesllm/morie)
"""Pre-trial custody credit days calculation (R v Summers, 1.5x)."""

from __future__ import annotations

import numpy as np

from morie.fn._containers import ESRes


def custody_days_credit(
    pretrial_days: np.ndarray,
    *,
    credit_ratio: float = 1.5,
) -> ESRes:
    """Pre-trial credit days at enhanced ratio (R v Summers, 2014).

    Parameters
    ----------
    pretrial_days : ndarray
        Days spent in pre-trial custody.
    credit_ratio : float
        Credit ratio (default 1.5).

    Returns
    -------
    ESRes
        estimate is mean credited days.
    """
    days = np.asarray(pretrial_days, dtype=float)
    credited = days * credit_ratio
    return ESRes(
        measure="custody_days_credit",
        estimate=float(np.mean(credited)),
        n=len(days),
        extra={
            "total_pretrial": float(np.sum(days)),
            "total_credited": float(np.sum(credited)),
            "credit_ratio": credit_ratio,
        },
    )


cstdy2 = custody_days_credit


def cheatsheet() -> str:
    return "custody_days_credit({}) -> Pre-trial custody credit days calculation (R v Summers, 1.5x"
