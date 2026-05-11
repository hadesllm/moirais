"""CentralTendency expression involving 'normally' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["cb_chapter_10_unnumbered_1"]


def cb_chapter_10_unnumbered_1(x):
    """
    CentralTendency expression involving 'normally' (auto-extracted; see reference for full context).

    Formula: on a simulated dataset of intelligence quotient (IQ) scores, which are normally distributed in the US population (μ = 100; SD = 15). We do this by

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
        See ``morie.fn.describe('cb10u1')`` for the full guide.

    References
    ----------
    Beginner's Guide to Statistics for Criminology and Criminal Justice using R (Wooditch et al., Springer 2021), ch.10 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression involving 'normally' (auto-extracted; see reference for full context).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression involving 'normally' (auto-extracted; see reference for full context)."},
    )


def cheatsheet():
    return "cb10u1: CentralTendency expression involving 'normally' (auto-extracted; see reference for full context)."
