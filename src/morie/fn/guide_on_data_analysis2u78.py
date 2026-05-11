"""PowerAndDesign expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_2_unnumbered_78"]


def guide_on_data_analysis_chapter_2_unnumbered_78(x):
    """
    PowerAndDesign expression (auto-extracted; see ref).

    Formula: [EQ] 𝑓(𝑥; 𝑣) =Γ((𝑣 + 1)/2)√𝑣𝜋Γ(𝑣/2)(1 +𝑥2

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
        See ``morie.fn.describe('guide_on_data_analysis2u78')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.2 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="PowerAndDesign expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "PowerAndDesign expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis2u78: PowerAndDesign expression (auto-extracted; see ref)."
