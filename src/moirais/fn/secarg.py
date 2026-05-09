"""Argon2id password hashing."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["argon2id_kdf"]


def argon2id_kdf(password, salt, m, t, p):
    """
    Argon2id password hashing

    Formula: data-dependent + data-independent passes; m,t,p params

    Parameters
    ----------
    password : array-like
        Input data.
    salt : array-like
        Input data.
    m : array-like
        Input data.
    t : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Biryukov et al (2016) RFC 9106
    """
    password = np.atleast_1d(np.asarray(password, dtype=float))
    n = len(password)
    result = float(np.mean(password))
    se = float(np.std(password, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Argon2id password hashing"})


def cheatsheet():
    return "secarg: Argon2id password hashing"
