"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_8_unnumbered_528"]


def blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_8_unnumbered_528(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: xi =pxi+1 +qxi1;

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
        See ``morie.fn.describe('blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit8u528')`` for the full guide.

    References
    ----------
    Blitzstein, Joseph K.  Hwang, Jessica - Introduction to probability, ch.8 (unnumbered)
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
    return "blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit8u528: Probability expression (auto-extracted; see ref)."
