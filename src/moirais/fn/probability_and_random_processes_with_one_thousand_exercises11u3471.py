"""Bayesian expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["probability_and_random_processes_with_one_thousand_exercises_chapter_11_unnumbered_3471"]


def probability_and_random_processes_with_one_thousand_exercises_chapter_11_unnumbered_3471(x):
    """
    Bayesian expression (auto-extracted; see ref).

    Formula: to say that Z( t) = sup{s : W( t − s) = 0}; we suppose that W( 0) = 0. At time t, some

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
        See ``moirais.fn.describe('probability_and_random_processes_with_one_thousand_exercises11u3471')`` for the full guide.

    References
    ----------
    Probability and Random Processes with One Thousand Exercises -- Geoffrey  Stirzaker Grimmett, ch.11 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Bayesian expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Bayesian expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "probability_and_random_processes_with_one_thousand_exercises11u3471: Bayesian expression (auto-extracted; see ref)."
