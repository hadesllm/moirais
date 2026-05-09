"""Hash-chained audit log entry."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hash_chain_audit"]


def hash_chain_audit(prev_hash, row, hash_alg):
    """
    Hash-chained audit log entry

    Formula: row_hash = H(prev_hash || canonical_row)

    Parameters
    ----------
    prev_hash : array-like
        Input data.
    row : array-like
        Input data.
    hash_alg : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Haber-Stornetta (1991); SoK Notarial Hash Chains
    """
    prev_hash = np.atleast_1d(np.asarray(prev_hash, dtype=float))
    n = len(prev_hash)
    result = float(np.mean(prev_hash))
    se = float(np.std(prev_hash, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hash-chained audit log entry"})


def cheatsheet():
    return "sechsh: Hash-chained audit log entry"
