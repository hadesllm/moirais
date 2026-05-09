"""Association expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_7_unnumbered_7"]


def edward_frenkel_love_and_math_the_heart_of_hidden_reality_chapter_7_unnumbered_7(x):
    """
    Association expression (auto-extracted; see ref).

    Formula: ax2 + bx + c = 0

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
        See ``moirais.fn.describe('edward_frenkel_love_and_math_the_heart_of_hidden_reality7u7')`` for the full guide.

    References
    ----------
    Edward Frenkel - Love and Math  The Heart of Hidden Reality, ch.7 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Association expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Association expression (auto-extracted; see ref)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Association expression (auto-extracted; see ref).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Association expression (auto-extracted; see ref).", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "edward_frenkel_love_and_math_the_heart_of_hidden_reality7u7: Association expression (auto-extracted; see ref)."
