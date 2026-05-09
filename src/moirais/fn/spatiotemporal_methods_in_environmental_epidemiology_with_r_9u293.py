"""CountModels expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_293"]


def spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_9_unnumbered_293(x):
    """
    CountModels expression (auto-extracted; see ref).

    Formula: αt = αt−1 + ωt , ωt∼ N(0,W ).

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
        See ``moirais.fn.describe('spatiotemporal_methods_in_environmental_epidemiology_with_r_9u293')`` for the full guide.

    References
    ----------
    spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818, ch.9 (unnumbered)
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
    return "spatiotemporal_methods_in_environmental_epidemiology_with_r_9u293: CountModels expression (auto-extracted; see ref)."
