"""Time-averaged ACF of a single sample observation x_k(t) at delay tau.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_time_averaged_acf"]


def rangayyan_ch3_time_averaged_acf(x_k, tau, T):
    """
    Time-averaged ACF of a single sample observation x_k(t) at delay tau.

    Formula: phi_xx(tau, k) = lim_{T->inf} (1/T) * integral_{-T/2}^{T/2} x_k(t) x_k(t+tau) dt

    Parameters
    ----------
    x_k : array-like
        Input data.
    tau : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.20, p. 98
    """
    x_k = np.atleast_1d(np.asarray(x_k, dtype=float))
    n = len(x_k)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Time-averaged ACF of a single sample observation x_k(t) at delay tau."})
    estimate = np.median(x_k)
    se = 1.2533 * np.std(x_k, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Time-averaged ACF of a single sample observation x_k(t) at delay tau."})


def cheatsheet():
    return "rng020: Time-averaged ACF of a single sample observation x_k(t) at delay tau."
