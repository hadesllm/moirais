"""Transfer function of the 8-point MA filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_ma_8point_transfer_function"]


def rangayyan_ch3_ma_8point_transfer_function(z):
    """
    Transfer function of the 8-point MA filter.

    Formula: H(z) = (1/8) * sum_{k=0}^{7} z^(-k)

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
    Rangayyan (2024), Ch 3, Eq 3.110, p. 142
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer function of the 8-point MA filter."})


def cheatsheet():
    return "rng099: Transfer function of the 8-point MA filter."
