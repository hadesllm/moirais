"""Probability equation extracted from Information theory MacKay.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["information_theory_mackay_chapter_3_equation_19"]


def information_theory_mackay_chapter_3_equation_19(x):
    """
    Probability equation extracted from Information theory MacKay.

    Formula: [EQ] P(sjF) =P(sjF;H1)P(H1) +P(sjF;H0)P(H0): (3.19)

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
        See ``moirais.fn.describe('information_theory_mackay3e19')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.3 eq.3.19
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Probability equation extracted from Information theory MacKay.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Probability equation extracted from Information theory MacKay."},
    )


def cheatsheet():
    return "information_theory_mackay3e19: Probability equation extracted from Information theory MacKay."
