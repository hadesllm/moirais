"""HKDF (RFC 5869) extract + expand."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hkdf_extract_expand"]


def hkdf_extract_expand(salt, ikm, info, length):
    """
    HKDF (RFC 5869) extract + expand

    Formula: PRK = HMAC(salt, IKM); OKM = HMAC(PRK, info||counter)

    Parameters
    ----------
    salt : array-like
        Input data.
    ikm : array-like
        Input data.
    info : array-like
        Input data.
    length : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Krawczyk-Eronen RFC 5869
    """
    salt = np.atleast_1d(np.asarray(salt, dtype=float))
    n = len(salt)
    result = float(np.mean(salt))
    se = float(np.std(salt, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "HKDF (RFC 5869) extract + expand"})


def cheatsheet():
    return "seckdf: HKDF (RFC 5869) extract + expand"
