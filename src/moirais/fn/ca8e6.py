"""Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["ca_chapter_8_equation_6"]


def ca_chapter_8_equation_6(x):
    """
    Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context).

    Formula: δ = r

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
        See ``moirais.fn.describe('ca8e6')`` for the full guide.

    References
    ----------
    Advanced Statistics in Criminology and Criminal Justice (Weisburd, Wilson, Wooditch & Britt, 5th ed, Springer 2022), ch.8 eq.8.6
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context).", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "ca8e6: Correlation expression involving 'noncentrality' (auto-extracted; see reference for full context)."
