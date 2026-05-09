"""Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_63(x):
    """
    Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.

    Formula: [EQ] Cov(X,Y ) = E(XY ) − µx µy = E(X )E(Y ) − µx µy = 0, (6.63)

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner6e63')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.6 eq.6.63
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."},
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e63: Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."
