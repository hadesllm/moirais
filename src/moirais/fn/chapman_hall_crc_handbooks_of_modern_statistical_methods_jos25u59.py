"""CausalInference expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_59"]


def chapman_hall_crc_handbooks_of_modern_statistical_methods_jos_chapter_25_unnumbered_59(x):
    """
    CausalInference expression (auto-extracted; see ref).

    Formula: 0 < λi = Pr ( Zi = 1 | xi) = Pr ( Zi = 1 | rTi , rCi, xi) < 1. Is there an unbiased estimate of δ? If

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
        See ``moirais.fn.describe('chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u59')`` for the full guide.

    References
    ----------
    [Chapman & Hall CRC Handbooks of Modern Statistical Methods] José R. Zubizarreta, Elizabeth A. Stuart, Dylan S. Small, Paul R - Handbook of Matching and Weighting Adjustments for Causal Inference, ch.25 (unnumbered)
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
    return "chapman_hall_crc_handbooks_of_modern_statistical_methods_jos25u59: CausalInference expression (auto-extracted; see ref)."
