"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_7"]


def advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th_chapter_13_unnumbered_7(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: [EQ] x = c(1, 2, 3, 1, 2), y = 5.1, z = letters[1.5])

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
        See ``moirais.fn.describe('advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u7')`` for the full guide.

    References
    ----------
    Advanced R (Second Edition) -- Hadley Wickham -- Chapman & Hall CRC the R, ch.13 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Probability expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Probability expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "advanced_r_second_edition_hadley_wickham_chapman_hall_crc_th13u7: Probability expression (auto-extracted; see ref)."
