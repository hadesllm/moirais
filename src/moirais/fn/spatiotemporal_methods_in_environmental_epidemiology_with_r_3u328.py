"""Spatial expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_328"]


def spatiotemporal_methods_in_environmental_epidemiology_with_r__chapter_3_unnumbered_328(x):
    """
    Spatial expression (auto-extracted; see ref).

    Formula: βt = (β1t , β2t , . . . ,βNSt ) is assumed to evolve as a first-order autoregressive process

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
        See ``moirais.fn.describe('spatiotemporal_methods_in_environmental_epidemiology_with_r_3u328')`` for the full guide.

    References
    ----------
    spatiotemporal-methods-in-environmental-epidemiology-with-r-chapman-amp-hall-crc-texts-in-statistical-science-2nbsped-1032397810-9781032397818, ch.3 (unnumbered)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Spatial expression (auto-extracted; see ref).",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Spatial expression (auto-extracted; see ref)."},
    )


def cheatsheet():
    return "spatiotemporal_methods_in_environmental_epidemiology_with_r_3u328: Spatial expression (auto-extracted; see ref)."
