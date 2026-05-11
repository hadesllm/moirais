# morie.fn — function file (hadesllm/morie)
"""ML-KEM-768 post-quantum key encapsulation (fn/ wrapper)."""

from __future__ import annotations

from ._containers import CryptoResult


def mlkem768_keygen() -> CryptoResult:
    """Generate ML-KEM-768 key pair.

    :return: CryptoResult with pk and sk in ``extra``.
    """
    from morie.crypto._mlkem import mlkem768_keygen as _keygen

    pk, sk = _keygen()
    return CryptoResult(
        algorithm="ML-KEM-768",
        operation="keygen",
        success=True,
        extra={"pk": pk, "sk": sk, "pk_len": len(pk), "sk_len": len(sk)},
    )


def mlkem768_encaps(pk: bytes) -> CryptoResult:
    """Encapsulate a shared secret.

    :param pk: Public key bytes.
    :return: CryptoResult with ct and shared_secret in ``extra``.
    """
    from morie.crypto._mlkem import mlkem768_encaps as _encaps

    ct, ss = _encaps(pk)
    return CryptoResult(
        algorithm="ML-KEM-768",
        operation="encaps",
        success=True,
        extra={"ct": ct, "shared_secret": ss, "ct_len": len(ct)},
    )


def mlkem768_decaps(sk: bytes, ct: bytes) -> CryptoResult:
    """Decapsulate the shared secret.

    :param sk: Secret key bytes.
    :param ct: Ciphertext bytes.
    :return: CryptoResult with shared_secret in ``extra``.
    """
    from morie.crypto._mlkem import mlkem768_decaps as _decaps

    ss = _decaps(sk, ct)
    return CryptoResult(
        algorithm="ML-KEM-768",
        operation="decaps",
        success=True,
        extra={"shared_secret": ss},
    )


mlkem = mlkem768_keygen


def cheatsheet() -> str:
    return "mlkem768_keygen({}) -> ML-KEM-768 post-quantum key encapsulation (fn/ wrapper)."
