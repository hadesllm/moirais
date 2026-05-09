"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_28_unnumbered_1403"]


def guide_on_data_analysis_chapter_28_unnumbered_1403(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: 𝑌𝑡 = 𝛽0 + 𝛽1𝑇𝑡 + 𝛽2𝐷𝑡 + 𝛽3(𝑇𝑡 × 𝐷𝑡) + 𝛽4𝑃𝑡⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟⏟

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
        See ``moirais.fn.describe('guide_on_data_analysis28u1403')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.28 (unnumbered)
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
    return "guide_on_data_analysis28u1403: Regression expression (auto-extracted; see ref)."
