"""CentralTendency equation extracted from Information theory MacKay.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["information_theory_mackay_chapter_19_equation_13"]


def information_theory_mackay_chapter_19_equation_13(x):
    """
    CentralTendency equation extracted from Information theory MacKay.

    Formula: 2=( + 2). The solution of this equation is

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
        See ``moirais.fn.describe('information_theory_mackay19e13')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.19 eq.19.13
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency equation extracted from Information theory MacKay.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency equation extracted from Information theory MacKay."},
    )


def cheatsheet():
    return "information_theory_mackay19e13: CentralTendency equation extracted from Information theory MacKay."
