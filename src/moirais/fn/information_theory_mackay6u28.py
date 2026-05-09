"""ContingencyTables expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["information_theory_mackay_chapter_6_unnumbered_28"]


def information_theory_mackay_chapter_6_unnumbered_28(x):
    """
    ContingencyTables expression (auto-extracted; see ref).

    Formula: fBN)2 = 0 has probability P(nB =1) = 0.4096; so the probability that z < 1

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
        See ``moirais.fn.describe('information_theory_mackay6u28')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.6 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="ContingencyTables expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "ContingencyTables expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "information_theory_mackay6u28: ContingencyTables expression (auto-extracted; see ref)."
