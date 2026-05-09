"""Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_10_equation_21(x):
    """
    Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: [EQ] ̂̄ 𝑧cregr = ̂̄ 𝑧𝜋 + ̂𝑏 ( ̄ 𝑥 −̂̄ 𝑥𝜋) , (10.21)

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
        See ``moirais.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r10e21')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.10 eq.10.21
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
    return "the_r_series_dick_j_brus_spatial_sampling_with_r10e21: Regression equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."
