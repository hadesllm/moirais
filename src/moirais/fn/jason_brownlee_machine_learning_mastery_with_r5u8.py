"""Spatial expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_8"]


def jason_brownlee_machine_learning_mastery_with_r_chapter_5_unnumbered_8(x):
    """
    Spatial expression (auto-extracted; see ref).

    Formula: preProc=c("BoxCox"), tuneGrid=grid, trControl=trainControl)

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
        See ``moirais.fn.describe('jason_brownlee_machine_learning_mastery_with_r5u8')`` for the full guide.

    References
    ----------
    Jason Brownlee - Machine Learning Mastery with R, ch.5 (unnumbered)
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
    return "jason_brownlee_machine_learning_mastery_with_r5u8: Spatial expression (auto-extracted; see ref)."
