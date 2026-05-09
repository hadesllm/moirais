"""CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_22(x):
    """
    CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.

    Formula: assume that the X, Y , and Z distributions in the Y = mX + Z relation are all known

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner6e22')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.6 eq.6.22
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."},
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e22: CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."
