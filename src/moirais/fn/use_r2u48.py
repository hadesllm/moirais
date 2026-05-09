"""Spatial expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["use_r_chapter_2_unnumbered_48"]


def use_r_chapter_2_unnumbered_48(x):
    """
    Spatial expression (auto-extracted; see ref).

    Formula: [+proj=utm +zone=13 +ellps=clrk66 +datum=NAD27 +units=m

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
        See ``moirais.fn.describe('use_r2u48')`` for the full guide.

    References
    ----------
    Use R, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Spatial expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Spatial expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "use_r2u48: Spatial expression (auto-extracted; see ref)."
