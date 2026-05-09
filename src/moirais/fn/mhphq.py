# moirais.fn — function file (hadesllm/moirais)
"""PHQ-9 depression screening score."""

import numpy as np

from ._containers import ESRes


def phq9_score(
    items: list | np.ndarray,
) -> ESRes:
    """Compute PHQ-9 (Patient Health Questionnaire-9) depression score.

    9 items scored 0-3. Total 0-27.
    Severity: 0-4 minimal, 5-9 mild, 10-14 moderate,
    15-19 moderately severe, 20-27 severe.

    Parameters
    ----------
    items : array-like
        Nine item responses, each in [0, 3].

    Returns
    -------
    ESRes

    References
    ----------
    Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001).
    The PHQ-9. Journal of General Internal Medicine, 16(9), 606-613.
    """
    a = np.asarray(items, dtype=int)
    if len(a) != 9:
        raise ValueError("PHQ-9 requires exactly 9 items")
    if np.any(a < 0) or np.any(a > 3):
        raise ValueError("Each item must be in [0, 3]")

    total = int(np.sum(a))
    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    elif total <= 19:
        severity = "moderately_severe"
    else:
        severity = "severe"

    return ESRes(
        measure="PHQ-9",
        estimate=float(total),
        extra={"severity": severity, "mdd_screen_positive": total >= 10},
    )


mhphq = phq9_score


def cheatsheet() -> str:
    return "phq9_score({}) -> PHQ-9 depression screening score."
