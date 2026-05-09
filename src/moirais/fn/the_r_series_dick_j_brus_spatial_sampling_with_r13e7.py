"""Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_7"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_7(x):
    """
    Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: equal size, 𝑤ℎ = 1/𝑛, and that 𝑛ℎ = 1. Therefore, Equation ( 13.5) reduces to

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
        See ``moirais.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r13e7')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.13 eq.13.7
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."},
    )


def cheatsheet():
    return "the_r_series_dick_j_brus_spatial_sampling_with_r13e7: Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."
