# moirais.fn — function file (hadesllm/moirais)
"""LDPC bit-flipping decoder."""

from __future__ import annotations

import numpy as np

from ._containers import CryptoResult


def ldpc_decode(H: np.ndarray, received: np.ndarray, max_iter: int = 50) -> CryptoResult:
    """Decode using LDPC bit-flipping algorithm.

    :param H: LDPC parity-check matrix (m x n) over GF(2).
    :param received: Received binary vector.
    :param max_iter: Maximum decoding iterations.
    :return: CryptoResult with decoded vector in ``extra``.
    """
    from moirais.crypto._ecc import ldpc_decode as _decode

    result = _decode(np.asarray(H), np.asarray(received), max_iter=max_iter)
    return CryptoResult(
        algorithm="LDPC",
        operation="decode",
        success=result["success"],
        extra=result,
    )


ldpcd = ldpc_decode


def cheatsheet() -> str:
    return "ldpc_decode({}) -> LDPC bit-flipping decoder."
