"""CountModels equation extracted from bivand2013.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["bivand2013_chapter_10_equation_3"]


def bivand2013_chapter_10_equation_3(x):
    """
    CountModels equation extracted from bivand2013.

    Formula: )O+−Oz

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
        See ``morie.fn.describe('bivand201310e3')`` for the full guide.

    References
    ----------
    bivand2013, ch.10 eq.10.3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CountModels equation extracted from bivand2013.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CountModels equation extracted from bivand2013."},
    )


def cheatsheet():
    return "bivand201310e3: CountModels equation extracted from bivand2013."
