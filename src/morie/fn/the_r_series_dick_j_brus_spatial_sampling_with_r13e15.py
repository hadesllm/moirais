"""Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_13_equation_15(x):
    """
    Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: [EQ] 𝑖𝑗] = ( ̂ 𝑧𝑖 − ̂ 𝑧𝑗)2 + 𝑆2(𝜖𝑖) + 𝑆2(𝜖𝑗) − 2𝑆2(𝜖𝑖, 𝜖𝑗) , (13.15)

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
        See ``morie.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r13e15')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.13 eq.13.15
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
    return "the_r_series_dick_j_brus_spatial_sampling_with_r13e15: Dispersion equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."
