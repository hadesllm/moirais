"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_24_unnumbered_1130"]


def guide_on_data_analysis_chapter_24_unnumbered_1130(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: [EQ] (𝛼𝛽)𝑎𝑗 = −(𝛼𝛽)1𝑗 − (𝛼𝛽)2𝑗 − ⋯ − (𝛼𝛽)𝑎−1,𝑗, 𝑗 = 1, … , 𝑏

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
        See ``morie.fn.describe('guide_on_data_analysis24u1130')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.24 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis24u1130: CentralTendency expression (auto-extracted; see ref)."
