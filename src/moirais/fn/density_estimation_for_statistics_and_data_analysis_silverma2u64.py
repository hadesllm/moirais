"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_64"]


def density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_64(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: where is the volume of the unit d-dimensional sphere: = 2, C2 = n,C 3 = 4;r/3, etc. We shall see in Section 4.4 that useful kernels for the case d = 2 are given by

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
        See ``moirais.fn.describe('density_estimation_for_statistics_and_data_analysis_silverma2u64')`` for the full guide.

    References
    ----------
    Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Probability expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Probability expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "density_estimation_for_statistics_and_data_analysis_silverma2u64: Probability expression (auto-extracted; see ref)."
