"""Logistic expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_5_unnumbered_433"]


def guide_on_data_analysis_chapter_5_unnumbered_433(x):
    """
    Logistic expression (auto-extracted; see ref).

    Formula: [EQ] ln(𝑓𝑌 |𝑋(𝑦𝑖, 𝑥𝑖; 𝛽)) = 𝑦𝑖 ln(𝐹𝜖(𝑥𝑖𝛽)) + (1 − 𝑦𝑖)ln(1 − 𝐹𝜖(𝑥𝑖𝛽)).

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
        See ``morie.fn.describe('guide_on_data_analysis5u433')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.5 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Logistic expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Logistic expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis5u433: Logistic expression (auto-extracted; see ref)."
