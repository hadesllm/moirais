"""CentralTendency equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.."""
import numpy as np
from scipy import stats

from ._richresult import RichResult, hypothesis_test_result

__all__ = ["andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_18"]


def andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp_chapter_6_equation_18(x):
    """
    CentralTendency equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.

    Formula: [EQ] λ(s) = λ0(s) exp{β + S(s)}, (6.18)

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
        See ``moirais.fn.describe('andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e18')`` for the full guide.

    References
    ----------
    Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling, ch.6 eq.6.18
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return RichResult(
        title="CentralTendency equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling.",
        summary_lines=[
            ("Estimate", result),
            ("Standard error", se),
            ("n", n),
        ],
        payload={"estimate": result, "se": se, "n": n, "method": "CentralTendency equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling."},
    )


def cheatsheet():
    return "andrew_b_lawson_using_r_for_bayesian_spatial_and_spatio_temp6e18: CentralTendency equation extracted from Andrew B Lawson - Using R for Bayesian Spatial and Spatio-Temporal Health Modeling."
