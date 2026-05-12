"""Winternitz One-Time Signature (WOTS+) -- verify."""

from __future__ import annotations

from ._containers import CryptoResult


def wots_verify(message: bytes, signature: list[bytes], pk: list[bytes], w: int = 16) -> CryptoResult:
    """Verify a WOTS+ signature.

    :param message: Original message bytes.
    :param signature: Signature from wots_sign().
    :param pk: Public key from wots_sign().
    :param w: Winternitz parameter.
    :return: CryptoResult with valid flag.
    """
    from morie.crypto._hashsig import wots_verify as _verify

    valid = _verify(message, signature, pk, w=w)
    return CryptoResult(
        algorithm="WOTS+",
        operation="verify",
        success=valid,
        extra={"valid": valid, "w": w},
    )


wotsv = wots_verify


def cheatsheet() -> str:
    return "wots_verify({}) -> Winternitz One-Time Signature (WOTS+) -- verify."
