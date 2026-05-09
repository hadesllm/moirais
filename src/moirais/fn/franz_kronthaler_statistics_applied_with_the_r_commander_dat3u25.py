"""MetaAnalysis expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_25"]


def franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_25(x):
    """
    MetaAnalysis expression (auto-extracted; see ref).

    Formula: [EQ] • H0: µ1 = µ2 = µ3 = µ4

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
        See ``moirais.fn.describe('franz_kronthaler_statistics_applied_with_the_r_commander_dat3u25')`` for the full guide.

    References
    ----------
    Franz Kronthaler - Statistics Applied with the R Commander  Data Analysis Is (Not) an Art-Springer (2024), ch.3 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="MetaAnalysis expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "MetaAnalysis expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "franz_kronthaler_statistics_applied_with_the_r_commander_dat3u25: MetaAnalysis expression (auto-extracted; see ref)."
