"""Regression expression involving 'understanding' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["cb_chapter_15_equation_3"]


def cb_chapter_15_equation_3(x):
    """
    Regression expression involving 'understanding' (auto-extracted; see reference for full context).

    Formula: yb bx residuals

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
        See ``moirais.fn.describe('cb15e3')`` for the full guide.

    References
    ----------
    Beginner's Guide to Statistics for Criminology and Criminal Justice using R (Wooditch et al., Springer 2021), ch.15 eq.15.3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Regression expression involving 'understanding' (auto-extracted; see reference for full context).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Regression expression involving 'understanding' (auto-extracted; see reference for full context)."},
    )


def cheatsheet():
    return "cb15e3: Regression expression involving 'understanding' (auto-extracted; see reference for full context)."
