"""Probability expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_18_unnumbered_53"]


def edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_18_unnumbered_53(x):
    """
    Probability expression (auto-extracted; see ref).

    Formula: permutation x1 /mapstoc〈a∇→x2 /mapstoc〈a∇→x3 /mapstoc〈a∇→x1, and its inverse.

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
        See ``morie.fn.describe('edward_frenkel_love_and_math_the_heart_of_hidden_reality18u53')`` for the full guide.

    References
    ----------
    Edward Frenkel - Love and Math  The Heart of Hidden Reality, ch.18 (unnumbered)
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
    return "edward_frenkel_love_and_math_the_heart_of_hidden_reality18u53: Probability expression (auto-extracted; see ref)."
