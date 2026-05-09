"""Spatial expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatial_data_analysis_with_r_chapter_7_unnumbered_25"]


def spatial_data_analysis_with_r_chapter_7_unnumbered_25(x):
    """
    Spatial expression (auto-extracted; see ref).

    Formula: plot(nb, coordinates(hh), col= 'red', lwd=2, add= TRUE)

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
        See ``moirais.fn.describe('spatial_data_analysis_with_r7u25')`` for the full guide.

    References
    ----------
    spatial data analysis with R, ch.7 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Spatial expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Spatial expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "spatial_data_analysis_with_r7u25: Spatial expression (auto-extracted; see ref)."
