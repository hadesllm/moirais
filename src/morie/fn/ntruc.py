# morie.fn — function file (hadesllm/morie)
"""NTRU encryption."""

from __future__ import annotations

from ._containers import CryptoResult


def ntru_encrypt(message: list[int], pk: list[int], n: int = 167, q: int = 128) -> CryptoResult:
    """Encrypt a message polynomial with NTRU.

    :param message: Message polynomial (coefficients in {-1, 0, 1}).
    :param pk: Public key polynomial.
    :param n: Polynomial degree.
    :param q: Modulus.
    :return: CryptoResult with ciphertext in ``extra``.
    """
    from morie.crypto._ntru import ntru_encrypt as _encrypt

    ct = _encrypt(message, pk, n=n, q=q)
    return CryptoResult(
        algorithm="NTRU",
        operation="encrypt",
        success=True,
        extra={"ciphertext": ct, "n": n, "q": q},
    )


ntruc = ntru_encrypt


def cheatsheet() -> str:
    return "ntru_encrypt({}) -> NTRU encryption."
