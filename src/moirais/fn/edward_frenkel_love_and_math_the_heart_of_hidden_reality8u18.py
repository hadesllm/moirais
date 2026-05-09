"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_8_unnumbered_18"]


def edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_8_unnumbered_18(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: finitely many factors in the above product. Let us denote the coefficient in front of qm by bm. So we have b1 = 1, b2 = −2, b3 = −1, b4 = 2, b5 = 1,

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
        See ``moirais.fn.describe('edward_frenkel_love_and_math_the_heart_of_hidden_reality8u18')`` for the full guide.

    References
    ----------
    Edward Frenkel - Love and Math  The Heart of Hidden Reality, ch.8 (unnumbered)
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
    return "edward_frenkel_love_and_math_the_heart_of_hidden_reality8u18: PowerAndDesign expression (auto-extracted; see ref)."
