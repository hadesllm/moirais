"""Bayesian expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_26"]


def andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_26(x):
    """
    Bayesian expression (auto-extracted; see ref).

    Formula: prior distribution, λk ∼ U(λk−1, λk+1) k = 1, ...., G. The classes are penalized

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
        See ``moirais.fn.describe('andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u26')`` for the full guide.

    References
    ----------
    Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling, ch.14 (unnumbered)
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
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u26: Bayesian expression (auto-extracted; see ref)."
