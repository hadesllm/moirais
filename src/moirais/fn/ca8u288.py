"""MetaAnalysis expression involving 'small' (auto-extracted; see reference for full context).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["ca_chapter_8_unnumbered_288"]


def ca_chapter_8_unnumbered_288(x1, x2):
    """
    MetaAnalysis expression involving 'small' (auto-extracted; see reference for full context).

    Formula: df = n –2 = 98. For a small effect size ( r = 0.1), the noncentrality parameter

    Parameters
    ----------
    x1 : array-like
        Input data.
    x2 : array-like
        Input data.

    Returns
    -------
    result : RichResult
        Inherits from ``dict`` (so ``isinstance(result, dict)`` is True
        and ``result["statistic"]`` / ``result.get(...)`` keep working),
        but also exposes a multi-section ``str(result)`` render. Keys: d.
        See ``moirais.fn.describe('ca8u288')`` for the full guide.

    References
    ----------
    Advanced Statistics in Criminology and Criminal Justice (Weisburd, Wilson, Wooditch & Britt, 5th ed, Springer 2022), ch.8 (unnumbered)
    """
    x1 = np.atleast_1d(np.asarray(x1, dtype=float))
    n = len(x1)
    result = float(np.mean(x1))
    se = float(np.std(x1, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="MetaAnalysis expression involving 'small' (auto-extracted; see reference for full context).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "MetaAnalysis expression involving 'small' (auto-extracted; see reference for full context)."},
    )


def cheatsheet():
    return "ca8u288: MetaAnalysis expression involving 'small' (auto-extracted; see reference for full context)."
