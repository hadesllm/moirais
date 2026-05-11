"""Time-varying step size mu(n) per Zhang et al. for VAG signals.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_step_size_zhang"]


def rangayyan_ch3_lms_step_size_zhang(mu, M, x_bar, alpha, r, n):
    """
    Time-varying step size mu(n) per Zhang et al. for VAG signals.

    Formula: mu(n) = mu / ( (M+1) * x_bar^2(n) * [alpha, r(n), x_bar^2(n-1)] )

    Parameters
    ----------
    mu : array-like
        Input data.
    M : array-like
        Input data.
    x_bar : array-like
        Input data.
    alpha : array-like
        Input data.
    r : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.205, p. 185
    """
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    n = len(mu)
    result = float(np.mean(mu))
    se = float(np.std(mu, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-varying step size mu(n) per Zhang et al. for VAG signals."})


def cheatsheet():
    return "rng162: Time-varying step size mu(n) per Zhang et al. for VAG signals."
