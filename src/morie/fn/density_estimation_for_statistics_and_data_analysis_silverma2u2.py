"""Nonparametric expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_2"]


def density_estimation_for_statistics_and_data_analysis_silverma_chapter_2_unnumbered_2(x):
    """
    Nonparametric expression (auto-extracted; see ref).

    Formula: [EQ] 1/(t) =nd^{t)l.< w )

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
        See ``morie.fn.describe('density_estimation_for_statistics_and_data_analysis_silverma2u2')`` for the full guide.

    References
    ----------
    Density estimation for statistics and data analysis -- Silverman, B  W -- Monographs on statistics and applied probability, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Nonparametric expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "density_estimation_for_statistics_and_data_analysis_silverma2u2: Nonparametric expression (auto-extracted; see ref)."
