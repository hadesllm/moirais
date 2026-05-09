# moirais.fn — function file (hadesllm/moirais)
"""Lamport one-time signature — sign."""

from __future__ import annotations

from ._containers import CryptoResult


def lamport_sign(message: bytes, sk: bytes | None = None) -> CryptoResult:
    """Sign a message with a Lamport one-time signature.

    If sk is None, generates a fresh key pair.

    :param message: Message bytes.
    :param sk: Private key (None to auto-generate).
    :return: CryptoResult with signature, pk, sk in ``extra``.
    """
    from moirais.crypto._hashsig import lamport_keygen
    from moirais.crypto._hashsig import lamport_sign as _sign

    sk_keys, pk_keys = lamport_keygen()
    sig = _sign(message, sk_keys)
    return CryptoResult(
        algorithm="Lamport-OTS",
        operation="sign",
        success=True,
        extra={
            "signature": sig,
            "pk": pk_keys,
            "sk": sk_keys,
            "n": len(sk_keys),
        },
    )


lamp = lamport_sign


def cheatsheet() -> str:
    return "lamport_sign({}) -> Lamport one-time signature — sign."
