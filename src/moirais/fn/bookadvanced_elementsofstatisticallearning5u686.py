"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_686"]


def bookadvanced_elementsofstatisticallearning_chapter_5_unnumbered_686(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: and often ˆγj = ¯yj, the mean of the yi falling in region Rj. For misclassification loss, ˆγj is the modal class of the observations falling in

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
        See ``moirais.fn.describe('bookadvanced_elementsofstatisticallearning5u686')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.5 (unnumbered)
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
    return "bookadvanced_elementsofstatisticallearning5u686: CentralTendency expression (auto-extracted; see ref)."
