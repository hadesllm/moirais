"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_24_unnumbered_1215"]


def guide_on_data_analysis_chapter_24_unnumbered_1215(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: 𝑌𝑖𝑗 = 𝜇. + 𝜏𝑖 + 𝛾1(𝑋𝑖𝑗1 − ̄𝑋..1) + 𝛾2(𝑋𝑖𝑗2 − ̄𝑋..2) + 𝜖𝑖𝑗

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
        See ``morie.fn.describe('guide_on_data_analysis24u1215')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.24 (unnumbered)
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
    return "guide_on_data_analysis24u1215: Regression expression (auto-extracted; see ref)."
