"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_29"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_11_unnumbered_29(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: The estimated temporal trend equals 0.0403 ∘C 𝑦−1, and the estimated standard error equals 0.0024794 ∘C 𝑦−1. Using t = 1.4 yields the estimated average

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
        See ``moirais.fn.describe('the_r_series_dick_j_brus_spatial_sampling_with_r11u29')`` for the full guide.

    References
    ----------
    [The R Series] Dick J. Brus - Spatial Sampling with R, ch.11 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "the_r_series_dick_j_brus_spatial_sampling_with_r11u29: CentralTendency expression (auto-extracted; see ref)."
