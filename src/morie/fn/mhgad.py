# morie.fn — function file (hadesllm/morie)
"""GAD-7 anxiety screening score."""

import numpy as np

from ._containers import ESRes


def gad7_score(
    items: list | np.ndarray,
) -> ESRes:
    """Compute GAD-7 (Generalised Anxiety Disorder-7) score.

    7 items scored 0-3. Total 0-21.
    Severity: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-21 severe.

    Parameters
    ----------
    items : array-like
        Seven responses, each in [0, 3].

    Returns
    -------
    ESRes

    References
    ----------
    Spitzer, R. L. et al. (2006). A brief measure for assessing GAD.
    Archives of Internal Medicine, 166(10), 1092-1097.
    """
    a = np.asarray(items, dtype=int)
    if len(a) != 7:
        raise ValueError("GAD-7 requires exactly 7 items")
    if np.any(a < 0) or np.any(a > 3):
        raise ValueError("Each item must be in [0, 3]")

    total = int(np.sum(a))
    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    else:
        severity = "severe"

    return ESRes(
        measure="GAD-7",
        estimate=float(total),
        extra={"severity": severity, "screen_positive": total >= 10},
    )


mhgad = gad7_score


def cheatsheet() -> str:
    return "gad7_score({}) -> GAD-7 anxiety screening score."
