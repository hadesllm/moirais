"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_917"]


def bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_917(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: [EQ] φ 2 = U (−π/ 4,π/ 4)

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
        See ``moirais.fn.describe('bookadvanced_elementsofstatisticallearning6u917')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.6 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning6u917: CentralTendency expression (auto-extracted; see ref)."
