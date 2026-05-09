"""Age-standardized rate (direct)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["age_standardize"]


def age_standardize(age_specific_rates, standard_pop):
    """
    Age-standardized rate (direct)

    Formula: sum w_i r_i where w = standard population weights

    Parameters
    ----------
    age_specific_rates : array-like
        Input data.
    standard_pop : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Boyle-Parkin (1991)
    """
    age_specific_rates = np.atleast_1d(np.asarray(age_specific_rates, dtype=float))
    n = len(age_specific_rates)
    result = float(np.mean(age_specific_rates))
    se = float(np.std(age_specific_rates, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Age-standardized rate (direct)"})


def cheatsheet():
    return "agstds: Age-standardized rate (direct)"
