"""CountModels expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bivand2013_chapter_7_unnumbered_241"]


def bivand2013_chapter_7_unnumbered_241(x):
    """
    CountModels expression (auto-extracted; see ref).

    Formula: with parametersν+Oi and α+Ei. In other words, the information provided

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
        See ``moirais.fn.describe('bivand20137u241')`` for the full guide.

    References
    ----------
    bivand2013, ch.7 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CountModels expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CountModels expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "bivand20137u241: CountModels expression (auto-extracted; see ref)."
