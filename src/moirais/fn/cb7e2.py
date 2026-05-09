"""CountModels expression involving 'verdicts' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["cb_chapter_7_equation_2"]


def cb_chapter_7_equation_2(x):
    """
    CountModels expression involving 'verdicts' (auto-extracted; see reference for full context).

    Formula: [EQ]  = −()

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
        See ``moirais.fn.describe('cb7e2')`` for the full guide.

    References
    ----------
    Beginner's Guide to Statistics for Criminology and Criminal Justice using R (Wooditch et al., Springer 2021), ch.7 eq.7.2
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CountModels expression involving 'verdicts' (auto-extracted; see reference for full context).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CountModels expression involving 'verdicts' (auto-extracted; see reference for full context)."},
    )


def cheatsheet():
    return "cb7e2: CountModels expression involving 'verdicts' (auto-extracted; see reference for full context)."
