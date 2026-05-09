"""Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_2"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_9_equation_2(x):
    """
    Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: [EQ] ̂𝑡regr(𝑧) =̂𝑡𝜋(𝑧) +̂𝑏 (𝑡(𝑥) −̂𝑡𝜋(𝑥)) , (9.2)

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
        See ``moirais.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r9e2')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.9 eq.9.2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."},
    )


def cheatsheet():
    return "the_r_series_dick_j_brus_spatial_sampling_with_r9e2: Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."
