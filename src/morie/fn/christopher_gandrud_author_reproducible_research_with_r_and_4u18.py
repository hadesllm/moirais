"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"]


def christopher_gandrud_author_reproducible_research_with_r_and__chapter_4_unnumbered_18(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: [EQ] <<Histogram>>=

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
        See ``morie.fn.describe('It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle')`` for the full guide.

    References
    ----------
    Christopher Gandrud (Author) - Reproducible Research with R and RStudio, ch.4 (unnumbered)
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
    return "It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"
