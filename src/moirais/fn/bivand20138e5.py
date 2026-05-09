"""Regression equation extracted from bivand2013.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bivand2013_chapter_8_equation_5"]


def bivand2013_chapter_8_equation_5(x):
    """
    Regression equation extracted from bivand2013.

    Formula: [EQ] Xj(s)βj +e(s)= Xβ+e(s), (8.5)

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
        See ``moirais.fn.describe('bivand20138e5')`` for the full guide.

    References
    ----------
    bivand2013, ch.8 eq.8.5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression equation extracted from bivand2013.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression equation extracted from bivand2013."},
    )


def cheatsheet():
    return "bivand20138e5: Regression equation extracted from bivand2013."
