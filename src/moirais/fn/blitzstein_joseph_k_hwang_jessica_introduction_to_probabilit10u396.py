"""CausalInference expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_396"]


def blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit_chapter_10_unnumbered_396(x):
    """
    CausalInference expression (auto-extracted; see ref).

    Formula: Hint: This problem relates to the previous one. Show thatP (Z = 1jS; X) =P (Z = 1jS),

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
        See ``moirais.fn.describe('blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u396')`` for the full guide.

    References
    ----------
    Blitzstein, Joseph K.  Hwang, Jessica - Introduction to probability, ch.10 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CausalInference expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CausalInference expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "blitzstein_joseph_k_hwang_jessica_introduction_to_probabilit10u396: CausalInference expression (auto-extracted; see ref)."
