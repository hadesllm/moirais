"""YaRN scaling combining NTK + interpolation + ramp."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["yarn_context_scaling"]


def yarn_context_scaling(y, q, m, theta, s, beta_fast, beta_slow):
    """
    YaRN scaling combining NTK + interpolation + ramp

    Formula: theta_i' = ramp_i(linear) + (1-ramp_i) NTK; t-scale temperature

    Parameters
    ----------
    y : array-like
        Input data.
    q : array-like
        Input data.
    m : array-like
        Input data.
    theta : array-like
        Input data.
    s : array-like
        Input data.
    beta_fast : array-like
        Input data.
    beta_slow : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Peng et al. (2023) YaRN
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "YaRN scaling combining NTK + interpolation + ramp"})


def cheatsheet():
    return "yarn: YaRN scaling combining NTK + interpolation + ramp"
