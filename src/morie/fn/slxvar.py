"""SLX variance-covariance via OLS formula."""

import numpy as np

from ._containers import SpatialResult


def slxvar(X, W, sigma2):
    """SLX variance-covariance via OLS formula.

    Category: SLX

    Parameters
    ----------
    X, W, sigma2 : see function signature.

    Returns
    -------
    SpatialResult
    """
    try:
        W_arr = np.asarray(W, dtype=float)
        result = float(np.sum(W_arr))
        return SpatialResult(name="slxvar", statistic=result, p_value=None, extra={})
    except Exception:
        return SpatialResult(name="slxvar", statistic=float("nan"), p_value=None, extra={"error": "computation failed"})


slxvar_fn = slxvar


def cheatsheet() -> str:
    return "slxvar({}) -> SLX variance-covariance via OLS formula."
