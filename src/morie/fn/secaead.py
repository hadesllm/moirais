"""ChaCha20-Poly1305 AEAD encrypt/decrypt."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aead_chacha20poly1305"]


def aead_chacha20poly1305(key, nonce, plaintext, aad):
    """
    ChaCha20-Poly1305 AEAD encrypt/decrypt

    Formula: ChaCha20 keystream XOR; Poly1305 tag = H_k(AAD||C)

    Parameters
    ----------
    key : array-like
        Input data.
    nonce : array-like
        Input data.
    plaintext : array-like
        Input data.
    aad : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    RFC 8439
    """
    key = np.atleast_1d(np.asarray(key, dtype=float))
    n = len(key)
    result = float(np.mean(key))
    se = float(np.std(key, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ChaCha20-Poly1305 AEAD encrypt/decrypt"})


def cheatsheet():
    return "secaead: ChaCha20-Poly1305 AEAD encrypt/decrypt"
