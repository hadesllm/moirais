"""Spatial equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3"]


def andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_3(x):
    """
    Spatial equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.

    Formula: [EQ] λ(s|ψ) = λ0(s|ψ0).λ1(s|ψ1). (6.3)

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
        See ``morie.fn.describe('andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3')`` for the full guide.

    References
    ----------
    Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling, ch.6 eq.6.3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="Spatial equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "Spatial equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling."},
    )


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e3: Spatial equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling."
