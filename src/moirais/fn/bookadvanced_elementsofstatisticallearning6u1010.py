"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1010"]


def bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1010(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: λ = 100 (35 df) wins when p = 100, and λ = 1000 (43 df) wins when

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
        See ``moirais.fn.describe('bookadvanced_elementsofstatisticallearning6u1010')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.6 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning6u1010: Regression expression (auto-extracted; see ref)."
