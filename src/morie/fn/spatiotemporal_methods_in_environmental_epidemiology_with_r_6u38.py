"""Dispersion expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_38"]


def spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_6_unnumbered_38(x):
    """
    Dispersion expression (auto-extracted; see ref).

    Formula: kXk)−1σ 2 + Aβk′β′

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
        See ``morie.fn.describe('spatiotemporal_methods_in_environmental_epidemiology_with_r_6u38')`` for the full guide.

    References
    ----------
    spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818, ch.6 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Dispersion expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Dispersion expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "spatiotemporal_methods_in_environmental_epidemiology_with_r_6u38: Dispersion expression (auto-extracted; see ref)."
