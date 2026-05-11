"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_113"]


def advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_25_unnumbered_113(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: to total = total + x[i] . Similar in-place operators are -=, *=, and /=.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: value.
        See ``morie.fn.describe('advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u113')`` for the full guide.

    References
    ----------
    Advanced R (Second Edition) -- Hadley Wickham -- Chapman & Hall CRC the R, ch.25 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th25u113: CentralTendency expression (auto-extracted; see ref)."
