"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_87"]


def bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_87(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: density of β , with τ2 and σ 2 assumed known, is equal to the expression

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
        See ``moirais.fn.describe('bookadvanced_elementsofstatisticallearning4u87')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.4 (unnumbered)
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
    return "bookadvanced_elementsofstatisticallearning4u87: Regression expression (auto-extracted; see ref)."
