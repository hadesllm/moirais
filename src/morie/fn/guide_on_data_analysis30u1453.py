"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_30_unnumbered_1453"]


def guide_on_data_analysis_chapter_30_unnumbered_1453(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: 𝑌𝑗𝑡 = 𝛽0 + 𝑁𝐽𝑗𝛽1 + 𝑃𝑂𝑆𝑇𝑡𝛽2 + (𝑁𝐽𝑗 × 𝑃𝑂𝑆𝑇𝑡)𝛽3 + 𝑋𝑗𝑡𝛽4 + 𝜖𝑗𝑡

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
        See ``morie.fn.describe('guide_on_data_analysis30u1453')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.30 (unnumbered)
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
    return "guide_on_data_analysis30u1453: Regression expression (auto-extracted; see ref)."
