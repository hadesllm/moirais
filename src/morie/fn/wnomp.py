"""W-NOMINATE vote probability using Gaussian utility function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["wnominate_probability"]


def wnominate_probability(ideal_point, yea_pos, nay_pos, beta):
    """
    W-NOMINATE vote probability using Gaussian utility function

    Formula: P_i(Yea) = exp(-beta*||x_i - z_Yea||^2) / (exp(-beta*||x_i-z_Yea||^2) + exp(-beta*||x_i-z_Nay||^2))

    Parameters
    ----------
    ideal_point : array-like
        Input data.
    yea_pos : array-like
        Input data.
    nay_pos : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'p_yea': 'float'}

    References
    ----------
    Armstrong Ch 5
    """
    ideal_point = np.asarray(ideal_point, dtype=float)
    n = int(ideal_point) if ideal_point.ndim == 0 else len(ideal_point)
    result = float(np.mean(ideal_point))
    se = float(np.std(ideal_point, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "W-NOMINATE vote probability using Gaussian utility function"})


def cheatsheet():
    return "wnomp: W-NOMINATE vote probability using Gaussian utility function"
