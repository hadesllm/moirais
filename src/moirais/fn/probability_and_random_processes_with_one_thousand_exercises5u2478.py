"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2478"]


def probability_and_random_processes_with_one_thousand_exercises_chapter_5_unnumbered_2478(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: A solution is a0 = ( − 1) j , ai = 0 for i ≥ 1, so that ˆXr+ k = ( − 1) k Xr , and the mean squared

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
        See ``moirais.fn.describe('probability_and_random_processes_with_one_thousand_exercises5u2478')`` for the full guide.

    References
    ----------
    Probability and Random Processes with One Thousand Exercises -- Geoffrey  Stirzaker Grimmett, ch.5 (unnumbered)
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
    return "probability_and_random_processes_with_one_thousand_exercises5u2478: Dispersion expression (auto-extracted; see ref)."
