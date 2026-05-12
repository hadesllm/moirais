"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["Look well into thyself; there is a source which will always spring up. -- Marcus Aurelius"]


def christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_21(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: print.xtable(m1_table, type = "html", caption.placement = "top")

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
        See ``morie.fn.describe('Look well into thyself; there is a source which will always spring up. -- Marcus Aurelius')`` for the full guide.

    References
    ----------
    Christopher Gandrud (Author) - Reproducible Research with R and RStudio, ch.4 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "Look well into thyself; there is a source which will always spring up. -- Marcus Aurelius"
