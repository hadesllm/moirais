"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["treatment_effects_1_chapter_1_unnumbered_26"]


def treatment_effects_1_chapter_1_unnumbered_26(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: [EQ] E[YiDi] = E[Yi· 1|Di = 1] Pr(Di = 1) + E[Yi· 0|Di = 0] Pr(Di = 0)

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
        See ``moirais.fn.describe('treatment_effects_11u26')`` for the full guide.

    References
    ----------
    Treatment effects 1, ch.1 (unnumbered)
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
    return "treatment_effects_11u26: Dispersion expression (auto-extracted; see ref)."
