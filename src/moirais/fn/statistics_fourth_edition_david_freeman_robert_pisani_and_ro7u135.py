"""Correlation expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_135"]


def statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_7_unnumbered_135(x):
    """
    Correlation expression (auto-extracted; see ref).

    Formula: 2. Left: ave x = 3.0, SD x = 1.0, ave y = 1.5, SD y = 0.5, positive correlation.

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
        See ``moirais.fn.describe('statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u135')`` for the full guide.

    References
    ----------
    Statistics, Fourth Edition -- David Freeman, Robert Pisani, and Roger Purves -- 2018, ch.7 (unnumbered)
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
    return "statistics_fourth_edition_david_freeman_robert_pisani_and_ro7u135: Correlation expression (auto-extracted; see ref)."
