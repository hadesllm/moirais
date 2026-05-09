"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_2_unnumbered_24"]


def guide_on_data_analysis_chapter_2_unnumbered_24(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: [EQ] 𝑃[𝐴1 ∪ 𝐴2 ∪ 𝐴3 … ] = 𝑃[𝐴1] + 𝑃[𝐴2] + 𝑃[𝐴3] + …

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
        See ``moirais.fn.describe('guide_on_data_analysis2u24')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.2 (unnumbered)
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
    return "guide_on_data_analysis2u24: Probability expression (auto-extracted; see ref)."
