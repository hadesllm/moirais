"""Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_power_cepstrum_sum"]


def rangayyan_ch4_power_cepstrum_sum(x_hat_p, h_hat_p, n):
    """
    Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected).

    Formula: y_hat_p(n) = x_hat_p(n) + h_hat_p(n)

    Parameters
    ----------
    x_hat_p : array-like
        Input data.
    h_hat_p : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.82, p. 251
    """
    x_hat_p = np.atleast_1d(np.asarray(x_hat_p, dtype=float))
    n = len(x_hat_p)
    result = float(np.mean(x_hat_p))
    se = float(np.std(x_hat_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected)."})


def cheatsheet():
    return "rng254: Power cepstrum of a convolution as sum of component power cepstra (cross-term neglected)."
