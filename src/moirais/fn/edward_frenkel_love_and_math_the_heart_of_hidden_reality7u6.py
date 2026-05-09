"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_7_unnumbered_6"]


def edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_7_unnumbered_6(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: Before Galois, mathematicians focused on trying to discover explicit formulas for solutions of equations like x2 = 2 and x3 − x + 1 = 0, called polynomial

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
        See ``moirais.fn.describe('edward_frenkel_love_and_math_the_heart_of_hidden_reality7u6')`` for the full guide.

    References
    ----------
    Edward Frenkel - Love and Math  The Heart of Hidden Reality, ch.7 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="PowerAndDesign expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "PowerAndDesign expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "edward_frenkel_love_and_math_the_heart_of_hidden_reality7u6: PowerAndDesign expression (auto-extracted; see ref)."
