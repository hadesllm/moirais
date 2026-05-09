"""Average output noise power of a matched filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_average_output_noise_power"]


def rangayyan_ch4_average_output_noise_power(P_eta_i, H, f):
    """
    Average output noise power of a matched filter.

    Formula: P_eta_o = (P_eta_i / 2) * integral_{-inf}^{inf} |H(f)|^2 df

    Parameters
    ----------
    P_eta_i : array-like
        Input data.
    H : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.37, p. 238
    """
    P_eta_i = np.atleast_1d(np.asarray(P_eta_i, dtype=float))
    n = len(P_eta_i)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average output noise power of a matched filter."})
    estimate = np.median(P_eta_i)
    se = 1.2533 * np.std(P_eta_i, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average output noise power of a matched filter."})


def cheatsheet():
    return "rng211: Average output noise power of a matched filter."
