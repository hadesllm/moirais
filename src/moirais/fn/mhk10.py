# moirais.fn — function file (hadesllm/moirais)
"""Kessler K10 psychological distress score."""

import numpy as np

from ._containers import ESRes


def k10_score(
    items: list | np.ndarray,
) -> ESRes:
    """Compute Kessler K10 psychological distress score.

    10 items scored 1-5. Total 10-50.
    Distress: 10-19 low, 20-24 mild, 25-29 moderate, 30-50 severe.

    Parameters
    ----------
    items : array-like
        Ten responses, each in [1, 5].

    Returns
    -------
    ESRes

    References
    ----------
    Kessler, R. C. et al. (2002). Short screening scales.
    Psychological Medicine, 32(6), 959-976.
    """
    a = np.asarray(items, dtype=int)
    if len(a) != 10:
        raise ValueError("K10 requires exactly 10 items")
    if np.any(a < 1) or np.any(a > 5):
        raise ValueError("Each item must be in [1, 5]")

    total = int(np.sum(a))
    if total <= 19:
        level = "low"
    elif total <= 24:
        level = "mild"
    elif total <= 29:
        level = "moderate"
    else:
        level = "severe"

    return ESRes(
        measure="K10",
        estimate=float(total),
        extra={"distress_level": level, "high_distress": total >= 22},
    )


mhk10 = k10_score


def cheatsheet() -> str:
    return "k10_score({}) -> Kessler K10 psychological distress score."
