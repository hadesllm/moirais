"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_unnumbered_445"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_unnumbered_445(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: between probability densities, such as ρx (x) and ρ(x, y), and actual probabilities, such as ρx (x) dx and ρ(x, y) dx d y. It can also get confusing if you don’t

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner5u445')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.5 (unnumbered)
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
    return "david_j_morin_probability_for_the_enthusiastic_beginner5u445: Probability expression (auto-extracted; see ref)."
