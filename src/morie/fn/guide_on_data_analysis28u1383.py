"""CausalInference expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["guide_on_data_analysis_chapter_28_unnumbered_1383"]


def guide_on_data_analysis_chapter_28_unnumbered_1383(x):
    """
    CausalInference expression (auto-extracted; see ref).

    Formula: 𝑌𝑡 = 𝛼0+𝛼1(𝑇𝑡−𝑇∗)+𝛼2(𝑇𝑡−𝑇∗)2+𝜏𝐷𝑡+𝛼3(𝑇𝑡−𝑇∗)𝐷𝑡+𝛼4(𝑇𝑡−𝑇∗)2𝐷𝑡+𝜖𝑡, for |𝑇𝑡−𝑇∗| < ℎ

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
        See ``morie.fn.describe('guide_on_data_analysis28u1383')`` for the full guide.

    References
    ----------
    guide on data analysis, ch.28 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CausalInference expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CausalInference expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "guide_on_data_analysis28u1383: CausalInference expression (auto-extracted; see ref)."
