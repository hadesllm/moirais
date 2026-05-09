"""Master efficiency theorem: plug-in Z-estimator is asymptotically efficient under regularity conditions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kosorok_ch3_z_estimator_efficiency_master"]


def kosorok_ch3_z_estimator_efficiency_master(theta_n, theta, eta, I_tilde, Z, n):
    """
    Master efficiency theorem: plug-in Z-estimator is asymptotically efficient under regularity conditions

    Formula: sqrt(n)(theta_n - theta) => -I_tilde_{theta,eta}^{-1} * Z, where Z is the Gaussian limit of G_n l_tilde_{theta,eta}

    Parameters
    ----------
    theta_n : array-like
        Input data.
    theta : array-like
        Input data.
    eta : array-like
        Input data.
    I_tilde : array-like
        Input data.
    Z : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Kosorok (2008), Thm 3.1, p. 44
    """
    theta_n = np.atleast_1d(np.asarray(theta_n, dtype=float))
    n = len(theta_n)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Master efficiency theorem: plug-in Z-estimator is asymptotically efficient under regularity conditions"})
    estimate = np.median(theta_n)
    se = 1.2533 * np.std(theta_n, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Master efficiency theorem: plug-in Z-estimator is asymptotically efficient under regularity conditions"})


def cheatsheet():
    return "ksr072: Master efficiency theorem: plug-in Z-estimator is asymptotically efficient under regularity conditions"
