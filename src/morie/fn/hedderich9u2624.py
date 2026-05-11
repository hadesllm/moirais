"""ContingencyTables expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["hedderich_chapter_9_unnumbered_2624"]


def hedderich_chapter_9_unnumbered_2624(x):
    """
    ContingencyTables expression (auto-extracted; see ref).

    Formula: ☞ 6; it is advantageous if it also holds: n1≈n2 orn2 ⪆√n1 forn1 > n2 (Sachs [Sac86a]). For

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
        See ``morie.fn.describe('hedderich9u2624')`` for the full guide.

    References
    ----------
    Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R, ch.9 (unnumbered)
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
    return "hedderich9u2624: ContingencyTables expression (auto-extracted; see ref)."
