"""XMSS extended Merkle signature scheme."""

from __future__ import annotations

from ._containers import CryptoResult


def xmss_sign(message: bytes, tree_height: int = 4, w: int = 16) -> CryptoResult:
    """Generate an XMSS key pair and sign a message.

    :param message: Message bytes to sign.
    :param tree_height: Merkle tree height (2^h one-time keys).
    :param w: WOTS Winternitz parameter.
    :return: CryptoResult with signature, public key, auth path.
    """
    from morie.crypto._hashsig import xmss_keygen
    from morie.crypto._hashsig import xmss_sign as _sign

    keys = xmss_keygen(tree_height=tree_height, w=w)
    sig = _sign(message, keys["sk"], keys["tree"])
    return CryptoResult(
        algorithm="XMSS",
        operation="sign",
        success=True,
        extra={
            "signature": sig,
            "pk": keys["pk"],
            "tree_height": tree_height,
            "w": w,
        },
    )


xmss = xmss_sign


def cheatsheet() -> str:
    return "xmss_sign({}) -> XMSS extended Merkle signature scheme."
