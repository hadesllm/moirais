"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatial_data_analysis_with_r_chapter_5_unnumbered_20"]


def spatial_data_analysis_with_r_chapter_5_unnumbered_20(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: [EQ] ## randomForest(x = dw[, 2:ncol(d)], y = dw[, "pa"], mtry = mt)

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
        See ``morie.fn.describe('spatial_data_analysis_with_r5u20')`` for the full guide.

    References
    ----------
    spatial data analysis with R, ch.5 (unnumbered)
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
    return "spatial_data_analysis_with_r5u20: Regression expression (auto-extracted; see ref)."
