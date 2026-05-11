"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_3"]


def franz_kronthaler_statistics_applied_with_the_r_commander_dat_chapter_3_unnumbered_3(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: 4.5. ra innovation = 9.0; sd innovation = 2.83; var innovation = 8.00; IQR innovation = 3.5; cv innovation = 56.67%.

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
        See ``morie.fn.describe('franz_kronthaler_statistics_applied_with_the_r_commander_dat3u3')`` for the full guide.

    References
    ----------
    Franz Kronthaler - Statistics Applied with the R Commander  Data Analysis Is (Not) an Art-Springer (2024), ch.3 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "franz_kronthaler_statistics_applied_with_the_r_commander_dat3u3: Dispersion expression (auto-extracted; see ref)."
