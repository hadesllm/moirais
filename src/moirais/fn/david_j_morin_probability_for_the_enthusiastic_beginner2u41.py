"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_41"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_unnumbered_41(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: This is indeed equal to E(X ) + E(Y ) = 1.5 + 2 = 3.5, as Eq. (3.7) claims.

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner2u41')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.2 (unnumbered)
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
    return "david_j_morin_probability_for_the_enthusiastic_beginner2u41: Probability expression (auto-extracted; see ref)."
