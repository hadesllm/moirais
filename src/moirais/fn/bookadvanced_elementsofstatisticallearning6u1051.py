"""Correlation expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1051"]


def bookadvanced_elementsofstatisticallearning_chapter_6_unnumbered_1051(x):
    """
    Correlation expression (auto-extracted; see ref).

    Formula: of the collection of tests is (1 − (1− α )M ). On the other hand, if the tests

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
        See ``moirais.fn.describe('bookadvanced_elementsofstatisticallearning6u1051')`` for the full guide.

    References
    ----------
    BookAdvanced elementsofstatisticallearning, ch.6 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation expression (auto-extracted; see ref)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation expression (auto-extracted; see ref).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Correlation expression (auto-extracted; see ref).", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning6u1051: Correlation expression (auto-extracted; see ref)."
