# morie.fn -- function file (hadesllm/morie)
"""ML-DSA (Dilithium) post-quantum signature -- verify."""

from __future__ import annotations

from ._containers import CryptoResult


def mldsa_verify(message: bytes, signature: bytes, pk: bytes) -> CryptoResult:
    """Verify an ML-DSA signature.

    :param message: Original message bytes.
    :param signature: Signature from mldsa_sign().
    :param pk: Public key from mldsa_keygen().
    :return: CryptoResult with valid flag.
    """
    from morie.crypto._dilithium import mldsa_verify as _verify

    valid = _verify(message, signature, pk)
    return CryptoResult(
        algorithm="ML-DSA",
        operation="verify",
        success=valid,
        extra={"valid": valid},
    )


mldsv = mldsa_verify


def cheatsheet() -> str:
    return "mldsa_verify({}) -> ML-DSA (Dilithium) post-quantum signature -- verify."
