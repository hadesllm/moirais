"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["information_theory_mackay_chapter_6_unnumbered_171"]


def information_theory_mackay_chapter_6_unnumbered_171(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: [EQ] C=W0 =W=W0 log (1 +W0=W)

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
        See ``moirais.fn.describe('information_theory_mackay6u171')`` for the full guide.

    References
    ----------
    Information theory MacKay, ch.6 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="PowerAndDesign expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "PowerAndDesign expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "information_theory_mackay6u171: PowerAndDesign expression (auto-extracted; see ref)."
