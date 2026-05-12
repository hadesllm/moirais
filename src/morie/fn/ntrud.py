# morie.fn -- function file (hadesllm/morie)
"""NTRU decryption."""

from __future__ import annotations

from ._containers import CryptoResult


def ntru_decrypt(ciphertext: list[int], sk: dict, n: int = 167, q: int = 128, p: int = 3) -> CryptoResult:
    """Decrypt an NTRU ciphertext.

    :param ciphertext: Ciphertext polynomial.
    :param sk: Secret key dict with f and fp.
    :param n: Polynomial degree.
    :param q: Large modulus.
    :param p: Small modulus.
    :return: CryptoResult with recovered message in ``extra``.
    """
    from morie.crypto._ntru import ntru_decrypt as _decrypt

    msg = _decrypt(ciphertext, sk, n=n, q=q, p=p)
    return CryptoResult(
        algorithm="NTRU",
        operation="decrypt",
        success=True,
        extra={"message": msg, "n": n, "q": q, "p": p},
    )


ntrud = ntru_decrypt


def cheatsheet() -> str:
    return "ntru_decrypt({}) -> NTRU decryption."
