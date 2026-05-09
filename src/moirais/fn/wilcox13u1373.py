"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["wilcox_chapter_13_unnumbered_1373"]


def wilcox_chapter_13_unnumbered_1373(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: ν = 38, reject. 7. W = 3.8, ν = 38, reject. 8. With equal sample sizes and equal sample

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
        See ``moirais.fn.describe('wilcox13u1373')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.13 (unnumbered)
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
    return "wilcox13u1373: PowerAndDesign expression (auto-extracted; see ref)."
