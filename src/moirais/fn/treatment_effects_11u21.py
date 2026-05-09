"""Regression expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["treatment_effects_1_chapter_1_unnumbered_21"]


def treatment_effects_1_chapter_1_unnumbered_21(x):
    """
    Regression expression (auto-extracted; see ref).

    Formula: ≡β0 +βiDi +Ui. (16)

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
        See ``moirais.fn.describe('treatment_effects_11u21')`` for the full guide.

    References
    ----------
    Treatment effects 1, ch.1 (unnumbered)
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
    return "treatment_effects_11u21: Regression expression (auto-extracted; see ref)."
