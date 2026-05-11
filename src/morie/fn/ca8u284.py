"""Inference expression involving 'noncentrality' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["ca_chapter_8_unnumbered_284"]


def ca_chapter_8_unnumbered_284(x):
    """
    Inference expression involving 'noncentrality' (auto-extracted; see reference for full context).

    Formula: 3( 0 . 12 × 300 = 3). Unlike the t-test where we could compute a tβ and

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
        See ``morie.fn.describe('ca8u284')`` for the full guide.

    References
    ----------
    Advanced Statistics in Criminology and Criminal Justice (Weisburd, Wilson, Wooditch & Britt, 5th ed, Springer 2022), ch.8 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Inference expression involving 'noncentrality' (auto-extracted; see reference for full context).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Inference expression involving 'noncentrality' (auto-extracted; see reference for full context)."},
    )


def cheatsheet():
    return "ca8u284: Inference expression involving 'noncentrality' (auto-extracted; see reference for full context)."
