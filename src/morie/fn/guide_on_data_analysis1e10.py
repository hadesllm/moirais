"""Regression equation extracted from guide on data analysis.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_1_equation_10"]


def guide_on_data_analysis_chapter_1_equation_10(x):
    """
    Regression equation extracted from guide on data analysis.

    Formula: rfe_result <- rfe(data[, -ncol(data)], data $y

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
        See ``morie.fn.describe('guide_on_data_analysis1e10')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.1 eq.1.10
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression equation extracted from guide on data analysis.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression equation extracted from guide on data analysis."},
    )


def cheatsheet():
    return "guide_on_data_analysis1e10: Regression equation extracted from guide on data analysis."
