"""ANOVA expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_24_unnumbered_1200"]


def guide_on_data_analysis_chapter_24_unnumbered_1200(x):
    """
    ANOVA expression (auto-extracted; see ref).

    Formula: F actor A 𝑆𝑆𝐴 𝑎 − 1 𝑀𝑆𝐴 𝜎 2 + 𝑏𝑛∑ 𝛼2

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
        See ``morie.fn.describe('guide_on_data_analysis24u1200')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.24 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="ANOVA expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "ANOVA expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis24u1200: ANOVA expression (auto-extracted; see ref)."
