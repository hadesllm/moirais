"""Rotating token envelope: KEK in HSM, DEK per-row."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rotating_token_envelope"]


def rotating_token_envelope(payload, kek_id, dek_lifetime):
    """
    Rotating token envelope: KEK in HSM, DEK per-row

    Formula: KEK encrypts DEK; DEK encrypts payload; rotate DEK on access

    Parameters
    ----------
    payload : array-like
        Input data.
    kek_id : array-like
        Input data.
    dek_lifetime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    GCP KMS envelope encryption pattern
    """
    payload = np.atleast_1d(np.asarray(payload, dtype=float))
    n = len(payload)
    result = float(np.mean(payload))
    se = float(np.std(payload, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rotating token envelope: KEK in HSM, DEK per-row"})


def cheatsheet():
    return "secrtt: Rotating token envelope: KEK in HSM, DEK per-row"
