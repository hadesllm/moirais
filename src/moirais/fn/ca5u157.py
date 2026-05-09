"""Logistic expression involving 'logit' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["ca_chapter_5_unnumbered_157"]


def ca_chapter_5_unnumbered_157(x):
    """
    Logistic expression involving 'logit' (auto-extracted; see reference for full context).

    Formula: = τ0,m − b1x1 − b2x2 − ... − bkxk Equation 5 :9

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
        See ``moirais.fn.describe('ca5u157')`` for the full guide.

    References
    ----------
    Advanced Statistics in Criminology and Criminal Justice (Weisburd, Wilson, Wooditch & Britt, 5th ed, Springer 2022), ch.5 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Logistic expression involving 'logit' (auto-extracted; see reference for full context).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Logistic expression involving 'logit' (auto-extracted; see reference for full context)."},
    )


def cheatsheet():
    return "ca5u157: Logistic expression involving 'logit' (auto-extracted; see reference for full context)."
