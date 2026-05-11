"""MetaAnalysis expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["hedderich_chapter_9_unnumbered_1926"]


def hedderich_chapter_9_unnumbered_1926(x):
    """
    MetaAnalysis expression (auto-extracted; see ref).

    Formula: rate is above the (k-fold) limit (usually at leastα=0.10). The probabilityβcharacterizes the consumer’s risk (buyer, purchaser) of mistakenly assuming that the limit for the defect rate is being

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
        See ``morie.fn.describe('hedderich9u1926')`` for the full guide.

    References
    ----------
    Hedderich, Sachs & Reynarowych (2023) Applied Statistics: Methods Using R, ch.9 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="MetaAnalysis expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "MetaAnalysis expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "hedderich9u1926: MetaAnalysis expression (auto-extracted; see ref)."
