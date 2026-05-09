"""Spatial expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bivand2013_chapter_2_unnumbered_59"]


def bivand2013_chapter_2_unnumbered_59(x):
    """
    Spatial expression (auto-extracted; see ref).

    Formula: [EQ] +to_meter=1.0]

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
        See ``moirais.fn.describe('bivand20132u59')`` for the full guide.

    References
    ----------
    bivand2013, ch.2 (unnumbered)
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
    return "bivand20132u59: Spatial expression (auto-extracted; see ref)."
