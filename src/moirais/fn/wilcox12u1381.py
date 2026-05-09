"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["wilcox_chapter_12_unnumbered_1381"]


def wilcox_chapter_12_unnumbered_1381(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: [EQ] 4/20 + 9 /20 = 6 .2, ˆν = 33, c = 2.99, reject. (d) (15 − 10)/

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
        See ``moirais.fn.describe('wilcox12u1381')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.12 (unnumbered)
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
    return "wilcox12u1381: PowerAndDesign expression (auto-extracted; see ref)."
