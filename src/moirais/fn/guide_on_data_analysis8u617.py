"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_8_unnumbered_617"]


def guide_on_data_analysis_chapter_8_unnumbered_617(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: 𝑌𝑖𝑗 = (𝛽0 + 𝑏1𝑖) + (𝛽1𝐿𝑖 + 𝛽2𝐻𝑖 + 𝛽3𝐶𝑖 + 𝑏2𝑖)𝑡𝑖𝑗 + 𝜖𝑖𝑗.

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
        See ``moirais.fn.describe('guide_on_data_analysis8u617')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.8 (unnumbered)
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
    return "guide_on_data_analysis8u617: Dispersion expression (auto-extracted; see ref)."
