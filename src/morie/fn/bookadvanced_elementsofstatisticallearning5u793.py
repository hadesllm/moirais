"""Nonparametric expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_793"]


def bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_793(x):
    """
    Nonparametric expression (auto-extracted; see ref).

    Formula: scale parameterγ for the radial kernel: Kγ (x,y ) = exp −γ||x − y||2. The optimal

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
        See ``morie.fn.describe('bookadvanced_elementsofstatisticallearning5u793')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.5 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Nonparametric expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning5u793: Nonparametric expression (auto-extracted; see ref)."
