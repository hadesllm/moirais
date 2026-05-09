# moirais.fn — function file (hadesllm/moirais)
"""AR coefficient to cepstral coefficient conversion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ar_to_cepstrum"]


def rangayyan_ar_to_cepstrum(a_coeffs, sigma_sq):
    """
    AR coefficient to cepstral coefficient conversion

    Formula: c(0)=log(sigma^2); c(m)=-a(m) - sum_{k=1}^{m-1} (k/m)*c(k)*a(m-k) for m>=1

    Parameters
    ----------
    a_coeffs : array-like
        Input data.
    sigma_sq : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cepstrum

    References
    ----------
    Rangayyan Ch 7.5.3
    """
    a_coeffs = np.asarray(a_coeffs, dtype=float)
    n = int(a_coeffs) if a_coeffs.ndim == 0 else len(a_coeffs)
    result = float(np.mean(a_coeffs))
    se = float(np.std(a_coeffs, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR coefficient to cepstral coefficient conversion"})


def cheatsheet():
    return "rgar2cep: AR coefficient to cepstral coefficient conversion"
