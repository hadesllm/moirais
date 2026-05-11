# morie.fn — function file (hadesllm/morie)
"""ML-DSA (Dilithium) post-quantum signature — keygen."""

from __future__ import annotations

from ._containers import CryptoResult


def mldsa_keygen() -> CryptoResult:
    """Generate an ML-DSA key pair.

    :return: CryptoResult with pk and sk in ``extra``.
    """
    from morie.crypto._dilithium import mldsa_keygen as _keygen

    pk, sk = _keygen()
    return CryptoResult(
        algorithm="ML-DSA",
        operation="keygen",
        success=True,
        extra={"pk": pk, "sk": sk, "pk_len": len(pk), "sk_len": len(sk)},
    )


mldsa = mldsa_keygen


def cheatsheet() -> str:
    return "mldsa_keygen({}) -> ML-DSA (Dilithium) post-quantum signature — keygen."
