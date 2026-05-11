"""CentralTendency equation extracted from Information theory MacKay.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["information_theory_mackay_chapter_24_equation_13"]


def information_theory_mackay_chapter_24_equation_13(x):
    """
    CentralTendency equation extracted from Information theory MacKay.

    Formula: 2 and 1=b0 =S=2 (see equation 24.13). This true posterior

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
        See ``morie.fn.describe('information_theory_mackay24e13')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.24 eq.24.13
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
    return "information_theory_mackay24e13: CentralTendency equation extracted from Information theory MacKay."
