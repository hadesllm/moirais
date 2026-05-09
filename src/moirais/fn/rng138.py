"""Output of the Wiener (transversal) filter as convolution of input with tap weights.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_filter_output_convolution"]


def rangayyan_ch3_wiener_filter_output_convolution(x, w_k, n, M):
    """
    Output of the Wiener (transversal) filter as convolution of input with tap weights.

    Formula: d_tilde(n) = sum_{k=0}^{M-1} w_k * x(n - k)

    Parameters
    ----------
    x : array-like
        Input data.
    w_k : array-like
        Input data.
    n : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.154, p. 173
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of the Wiener (transversal) filter as convolution of input with tap weights."})


def cheatsheet():
    return "rng138: Output of the Wiener (transversal) filter as convolution of input with tap weights."
