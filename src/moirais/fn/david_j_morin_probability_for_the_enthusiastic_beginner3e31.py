"""CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_31(x):
    """
    CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.

    Formula: [EQ] V ar(X1 + X2 + · · · + Xn ) = V ar(X1 + X2 + · · · + Xn−1) + V ar(Xn ). (3.31)

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner3e31')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.3 eq.3.31
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
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e31: CentralTendency equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."
