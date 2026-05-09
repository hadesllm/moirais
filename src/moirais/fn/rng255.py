"""Relation between power cepstrum and complex cepstrum.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_power_cepstrum_relation_to_complex"]


def rangayyan_ch4_power_cepstrum_relation_to_complex(y_hat, n):
    """
    Relation between power cepstrum and complex cepstrum.

    Formula: y_hat_p(n) = [ y_hat(n) + y_hat(-n) ]^2

    Parameters
    ----------
    y_hat : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.83, p. 251
    """
    y_hat = np.atleast_1d(np.asarray(y_hat, dtype=float))
    n = len(y_hat)
    result = float(np.mean(y_hat))
    se = float(np.std(y_hat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Relation between power cepstrum and complex cepstrum."})


def cheatsheet():
    return "rng255: Relation between power cepstrum and complex cepstrum."
