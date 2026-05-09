"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_368"]


def spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_13_unnumbered_368(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: A health count, Y , is Poisson distributed with mean exp{α0 + α1z} conditional on an exposure Z = z with α0 = 0 and α1 = 1, which are not known

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
        See ``moirais.fn.describe('spatiotemporal_methods_in_environmental_epidemiology_with_r_13u368')`` for the full guide.

    References
    ----------
    spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818, ch.13 (unnumbered)
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
    return "spatiotemporal_methods_in_environmental_epidemiology_with_r_13u368: CentralTendency expression (auto-extracted; see ref)."
