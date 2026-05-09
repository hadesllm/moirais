"""Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_16"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_16(x):
    """
    Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.

    Formula: m2σ2x + σ2z and r = mσx√

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
        See ``moirais.fn.describe('david_j_morin_probability_for_the_enthusiastic_beginner6e16')`` for the full guide.

    References
    ----------
    David J. Morin - Probability  For the Enthusiastic Beginner, ch.6 eq.6.16
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner.", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "david_j_morin_probability_for_the_enthusiastic_beginner6e16: Correlation equation extracted from David J. Morin - Probability  For the Enthusiastic Beginner."
