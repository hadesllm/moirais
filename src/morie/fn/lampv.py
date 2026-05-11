# morie.fn — function file (hadesllm/morie)
"""Lamport one-time signature — verify."""

from __future__ import annotations

from ._containers import CryptoResult


def lamport_verify(message: bytes, signature: list[bytes], pk: list[list[bytes]]) -> CryptoResult:
    """Verify a Lamport one-time signature.

    :param message: Original message bytes.
    :param signature: Signature from lamport_sign().
    :param pk: Public key from lamport_sign().
    :return: CryptoResult with valid flag.
    """
    from morie.crypto._hashsig import lamport_verify as _verify

    valid = _verify(message, signature, pk)
    return CryptoResult(
        algorithm="Lamport-OTS",
        operation="verify",
        success=valid,
        extra={"valid": valid},
    )


lampv = lamport_verify


def cheatsheet() -> str:
    return "lamport_verify({}) -> Lamport one-time signature — verify."
