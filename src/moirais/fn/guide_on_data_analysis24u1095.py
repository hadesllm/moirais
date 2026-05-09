"""ANOVA expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_24_unnumbered_1095"]


def guide_on_data_analysis_chapter_24_unnumbered_1095(x):
    """
    ANOVA expression (auto-extracted; see ref).

    Formula: 2. Model 2: 𝑌𝑖𝑗 = 𝜇 + 𝜏𝑖 + 𝜖𝑖𝑗 where ∑𝑖 𝜏𝑖 = 0.

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
        See ``moirais.fn.describe('guide_on_data_analysis24u1095')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.24 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="ANOVA expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "ANOVA expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis24u1095: ANOVA expression (auto-extracted; see ref)."
