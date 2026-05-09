"""CountModels expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_30_unnumbered_1644"]


def guide_on_data_analysis_chapter_30_unnumbered_1644(x):
    """
    CountModels expression (auto-extracted; see ref).

    Formula: 𝑌𝑖𝑡 = exp(𝛽0 + 𝛽1𝐷𝑖 × 𝑃𝑜𝑠𝑡𝑡 + 𝛽2𝐷𝑖 + 𝛽3𝑃𝑜𝑠𝑡𝑡 + 𝑋𝑖𝑡)𝜖𝑖𝑡

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
        See ``moirais.fn.describe('guide_on_data_analysis30u1644')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.30 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CountModels expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CountModels expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis30u1644: CountModels expression (auto-extracted; see ref)."
