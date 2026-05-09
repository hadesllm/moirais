"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_489"]


def blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_489(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: 4. If the coin lands Heads, accept the proposal and set n+1 =x0. Otherwise, stay in place and set n+1 =x.

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
        See ``moirais.fn.describe('blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u489')`` for the full guide.

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
    return "blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u489: Probability expression (auto-extracted; see ref)."
