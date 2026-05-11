"""Correlation expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u_chapter_2_unnumbered_1"]


def benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u_chapter_2_unnumbered_1(x):
    """
    Correlation expression (auto-extracted; see ref).

    Formula: as psychology and education, r = .10 is generally considered a small relationship, r = .30 is generally considered of medium size, and r = .50 is large,3 but

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
        See ``morie.fn.describe('benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u2u1')`` for the full guide.

    References
    ----------
    Benjamin J. Lovett - Practical Psychometrics  A Guide for Test Users, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = min(len(x), len(y))
    if n < 3:
        return hypothesis_test_result(
            test_name="Correlation expression (auto-extracted; see ref).",
            statistic=float("nan"),
            pvalue=float("nan"),
            warnings=["n<3: insufficient pairs for correlation."],
            extra_summary=[("n", n)],
            extra_payload={"n": n, "method": "Correlation expression (auto-extracted; see ref)."},
        )
    result = stats.spearmanr(x[:n], y[:n])
    return hypothesis_test_result(
        test_name="Correlation expression (auto-extracted; see ref).",
        statistic=float(result.statistic),
        pvalue=float(result.pvalue),
        extra_summary=[("n", n)],
        extra_payload={"n": n, "method": "Correlation expression (auto-extracted; see ref).", "p_value": float(result.pvalue)},
    )


def cheatsheet():
    return "benjamin_j_lovett_practical_psychometrics_a_guide_for_test_u2u1: Correlation expression (auto-extracted; see ref)."
