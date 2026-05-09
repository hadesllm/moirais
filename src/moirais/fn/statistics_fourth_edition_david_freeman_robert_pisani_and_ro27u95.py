"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_27_unnumbered_95"]


def statistics_fourth_edition_david_freeman_robert_pisani_and_ro_chapter_27_unnumbered_95(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: [EQ] Z = ( ˆp − ˆq)/√

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
        See ``moirais.fn.describe('statistics_fourth_edition_david_freeman_robert_pisani_and_ro27u95')`` for the full guide.

    References
    ----------
    Statistics, Fourth Edition -- David Freeman, Robert Pisani, and Roger Purves -- 2018, ch.27 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "statistics_fourth_edition_david_freeman_robert_pisani_and_ro27u95: Dispersion expression (auto-extracted; see ref)."
