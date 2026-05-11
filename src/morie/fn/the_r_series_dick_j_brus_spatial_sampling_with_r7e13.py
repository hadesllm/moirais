"""CentralTendency equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_13"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_7_equation_13(x):
    """
    CentralTendency equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: inclusion probabilities of the PSUs then equal 𝜋𝑗 = 𝑛/𝑁 , 𝑗 = 1, … , 𝑁 , and

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
        See ``morie.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r7e13')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.7 eq.7.13
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."},
    )


def cheatsheet():
    return "the_r_series_dick_j_brus_spatial_sampling_with_r7e13: CentralTendency equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."
