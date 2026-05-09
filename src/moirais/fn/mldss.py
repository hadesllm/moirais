# moirais.fn — function file (hadesllm/moirais)
"""ML-DSA (Dilithium) post-quantum signature — sign."""

from __future__ import annotations

from ._containers import CryptoResult


def mldsa_sign(message: bytes, sk: bytes) -> CryptoResult:
    """Sign a message with ML-DSA.

    :param message: Message bytes.
    :param sk: Secret key from mldsa_keygen().
    :return: CryptoResult with signature in ``extra``.
    """
    from moirais.crypto._dilithium import mldsa_sign as _sign

    sig = _sign(message, sk)
    return CryptoResult(
        algorithm="ML-DSA",
        operation="sign",
        success=True,
        extra={"signature": sig, "sig_len": len(sig)},
    )


mldss = mldsa_sign


def cheatsheet() -> str:
    return "mldsa_sign({}) -> ML-DSA (Dilithium) post-quantum signature — sign."
