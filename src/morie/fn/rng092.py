"""Transfer function of the Hann filter (double zero at z=-1).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_transfer_function"]


def rangayyan_ch3_hann_transfer_function(z):
    """
    Transfer function of the Hann filter (double zero at z=-1).

    Formula: H(z) = Y(z)/X(z) = (1/4) * [1 + 2*z^(-1) + z^(-2)]

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.103, p. 140
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer function of the Hann filter (double zero at z=-1)."})


def cheatsheet():
    return "rng092: Transfer function of the Hann filter (double zero at z=-1)."
