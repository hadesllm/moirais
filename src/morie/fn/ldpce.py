# morie.fn — function file (hadesllm/morie)
"""LDPC encoding."""

from __future__ import annotations

import numpy as np

from ._containers import CryptoResult


def ldpc_encode(G: np.ndarray, message: np.ndarray) -> CryptoResult:
    """Encode a message using a generator matrix.

    :param G: Generator matrix (k x n) over GF(2).
    :param message: Binary message vector (length k).
    :return: CryptoResult with codeword in ``extra``.
    """
    from morie.crypto._ecc import ldpc_encode as _encode

    result = _encode(np.asarray(G), np.asarray(message))
    return CryptoResult(
        algorithm="LDPC",
        operation="encode",
        success=True,
        extra=result,
    )


ldpce = ldpc_encode


def cheatsheet() -> str:
    return "ldpc_encode({}) -> LDPC encoding."
