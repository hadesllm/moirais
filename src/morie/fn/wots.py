"""Winternitz One-Time Signature (WOTS+) — sign."""

from __future__ import annotations

from ._containers import CryptoResult


def wots_sign(message: bytes, sk: bytes | None = None, w: int = 16) -> CryptoResult:
    """Sign a message with WOTS+.

    If sk is None, generates a fresh key pair.

    :param message: Message bytes.
    :param sk: Private key (None to auto-generate).
    :param w: Winternitz parameter (chain length).
    :return: CryptoResult with signature, pk, sk in ``extra``.
    """
    from morie.crypto._hashsig import wots_keygen
    from morie.crypto._hashsig import wots_sign as _sign

    sk_keys, pk_keys = wots_keygen(w=w)
    sig = _sign(message, sk_keys, w=w)
    return CryptoResult(
        algorithm="WOTS+",
        operation="sign",
        success=True,
        extra={
            "signature": sig,
            "pk": pk_keys,
            "sk": sk_keys,
            "w": w,
        },
    )


wots = wots_sign


def cheatsheet() -> str:
    return "wots_sign({}) -> Winternitz One-Time Signature (WOTS+) — sign."
