"""Expected log predictive density (elpd)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["expected_log_predictive_density"]


def expected_log_predictive_density(log_lik):
    """
    Expected log predictive density (elpd)

    Formula: elpd = E[log p(y_tilde)]

    Parameters
    ----------
    log_lik : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vehtari, Gelman, Gabry (2017)
    """
    log_lik = np.atleast_1d(np.asarray(log_lik, dtype=float))
    n = len(log_lik)
    result = float(np.mean(log_lik))
    se = float(np.std(log_lik, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expected log predictive density (elpd)"})


def cheatsheet():
    return "elpdc: Expected log predictive density (elpd)"
