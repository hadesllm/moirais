"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_468"]


def bookadvanced_elementsofstatisticallearning_chapter_4_unnumbered_468(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: • If in addition σ m = σ > 0 is fixed, and M ↑ N , then the maximum likelihood estimate for (6.32) approaches the kernel d ensity

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
        See ``moirais.fn.describe('bookadvanced_elementsofstatisticallearning4u468')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.4 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning4u468: Dispersion expression (auto-extracted; see ref)."
