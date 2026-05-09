"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["wilcox_chapter_15_unnumbered_1389"]


def wilcox_chapter_15_unnumbered_1389(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: ˆσ δ = 0.0536, 0.95 confidence interval is ( −0.29 , −0.085), reject. 3.X 2 = 7.43, reject. No, it is

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
        See ``moirais.fn.describe('wilcox15u1389')`` for the full guide.

    References
    ----------
    Wilcox, R.R. (2017) Modern Statistics for the Social and Behavioral Sciences, ch.15 (unnumbered)
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
    return "wilcox15u1389: CentralTendency expression (auto-extracted; see ref)."
