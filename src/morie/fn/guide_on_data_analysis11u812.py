"""Bayesian expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_11_unnumbered_812"]


def guide_on_data_analysis_chapter_11_unnumbered_812(x):
    """
    Bayesian expression (auto-extracted; see ref).

    Formula: 𝑌𝑖𝑗𝑘𝑡 = 𝑌𝑖𝑗𝑘𝑡−1 + 𝑋𝑖𝑡𝛽 + 𝑇𝑖𝑡𝜏𝑗 + (𝑊𝑖 + 𝑃𝑘 + 𝜖𝑖𝑗𝑘𝑡)

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
        See ``morie.fn.describe('guide_on_data_analysis11u812')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.11 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Bayesian expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Bayesian expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis11u812: Bayesian expression (auto-extracted; see ref)."
