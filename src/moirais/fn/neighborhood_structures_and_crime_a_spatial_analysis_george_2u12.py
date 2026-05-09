"""Spatial expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_12"]


def neighborhood_structures_and_crime_a_spatial_analysis_george__chapter_2_unnumbered_12(x):
    """
    Spatial expression (auto-extracted; see ref).

    Formula: βµ γ=+

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
        See ``moirais.fn.describe('neighborhood_structures_and_crime_a_spatial_analysis_george_2u12')`` for the full guide.

    References
    ----------
    Neighborhood Structures And Crime A Spatial Analysis George Kikuchi, ch.2 (unnumbered)
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
    return "neighborhood_structures_and_crime_a_spatial_analysis_george_2u12: Spatial expression (auto-extracted; see ref)."
