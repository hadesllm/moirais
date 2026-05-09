"""Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["ca_chapter_11_equation_31"]


def ca_chapter_11_equation_31(x):
    """
    Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context).

    Formula: d2 + n1+n2

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
        See ``moirais.fn.describe('ca11e31')`` for the full guide.

    References
    ----------
    Advanced Statistics in Criminology and Criminal Justice (Weisburd, Wilson, Wooditch & Britt, 5th ed, Springer 2022), ch.11 eq.11.31
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context).", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "ca11e31: Correlation expression involving 'dffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffiffi' (auto-extracted; see reference for full context)."
