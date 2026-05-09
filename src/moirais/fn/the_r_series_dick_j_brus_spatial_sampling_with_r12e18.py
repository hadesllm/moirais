"""PowerAndDesign equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_18"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_18(x):
    """
    PowerAndDesign equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.

    Formula: [EQ] 𝑓(𝜃|z, 𝑛)d𝜃 = 1 − 𝛼 , (12.18)

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
        See ``moirais.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r12e18')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.12 eq.12.18
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="PowerAndDesign equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "PowerAndDesign equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."},
    )


def cheatsheet():
    return "the_r_series_dick_j_brus_spatial_sampling_with_r12e18: PowerAndDesign equation extracted from [The R Series] Dick J. Brus - Spatial Sampling with R."
