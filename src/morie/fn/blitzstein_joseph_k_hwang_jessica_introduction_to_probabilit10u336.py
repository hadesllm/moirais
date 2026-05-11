"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_336"]


def blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_336(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: [EQ] E(W ) = (E(VjV  3b=2)b)P (V  3b=2) = (3b=4b) (3b=2) =3b2=8:

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
        See ``morie.fn.describe('blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u336')`` for the full guide.

    References
    ----------
    Blitzstein, Joseph K.  Hwang, Jessica - Introduction to probability, ch.10 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Probability expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Probability expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u336: Probability expression (auto-extracted; see ref)."
