"""Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_58(x):
    """
    Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.

    Formula: σsingle = 0.37, σ tot = 37, σ avg = 0.0037. (3.58)

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner3e58')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.3 eq.3.58
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
    return "david_j_morin_probability_for_the_enthusiastic_beginner3e58: Dispersion equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."
