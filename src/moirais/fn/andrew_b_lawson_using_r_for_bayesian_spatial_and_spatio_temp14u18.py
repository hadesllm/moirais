"""CentralTendency expression (auto-extracted; see ref).."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_18"]


def andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_14_unnumbered_18(x):
    """
    CentralTendency expression (auto-extracted; see ref).

    Formula: it is possible to assume log( θij) = α0 + Si + Tj + xt

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
        See ``moirais.fn.describe('andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u18')`` for the full guide.

    References
    ----------
    Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling, ch.14 (unnumbered)
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
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp14u18: CentralTendency expression (auto-extracted; see ref)."
