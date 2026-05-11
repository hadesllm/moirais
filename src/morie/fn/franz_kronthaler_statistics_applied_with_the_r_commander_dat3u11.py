"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_11"]


def franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_11(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: [EQ] = 22 × 1.642 × 0.25(1 − 0.25)

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
        See ``morie.fn.describe('franz_kronthaler_statistics_applied_with_the_r_commander_dat3u11')`` for the full guide.

    References
    ----------
    Franz Kronthaler - Statistics Applied with the R Commander  Data Analysis Is (Not) an Art-Springer (2024), ch.3 (unnumbered)
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
    return "franz_kronthaler_statistics_applied_with_the_r_commander_dat3u11: CentralTendency expression (auto-extracted; see ref)."
