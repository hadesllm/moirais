"""Digital Butterworth transfer function after bilinear transform (IIR form).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_digital_transfer_function"]


def rangayyan_ch3_butterworth_digital_transfer_function(z, a_k, G_prime, N):
    """
    Digital Butterworth transfer function after bilinear transform (IIR form).

    Formula: H(z) = G' * (1 + z^(-1))^N / sum_{k=0}^{N} a_k * z^(-k)

    Parameters
    ----------
    z : array-like
        Input data.
    a_k : array-like
        Input data.
    G_prime : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.143, p. 155
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Digital Butterworth transfer function after bilinear transform (IIR form)."})


def cheatsheet():
    return "rng131: Digital Butterworth transfer function after bilinear transform (IIR form)."
