"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_116"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_116(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: [EQ] (1/2) · (1)2 + (1/2) · (−1)2 = 1. (3.103)

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner2u116')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner2u116: Dispersion expression (auto-extracted; see ref)."
