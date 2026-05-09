# moirais.fn — function file (hadesllm/moirais)
"""Classic McEliece syndrome-based encryption."""

from __future__ import annotations

import numpy as np

from ._containers import CryptoResult


def mceliece_encrypt(message: bytes, pk: np.ndarray | None = None) -> CryptoResult:
    """Generate keys and encapsulate with McEliece.

    If pk is None, generates a fresh key pair first.

    :param message: Ignored (KEM mode — encapsulates a random key).
    :param pk: Public key matrix (None to auto-generate).
    :return: CryptoResult with shared_secret, syndrome, keys in ``extra``.
    """
    from moirais.crypto._mceliece import mceliece_encaps, mceliece_keygen

    keys = mceliece_keygen()
    pub = keys["pk"]
    ss, syndrome = mceliece_encaps(pub)
    return CryptoResult(
        algorithm="McEliece",
        operation="encapsulate",
        success=True,
        extra={
            "shared_secret": ss,
            "syndrome": syndrome,
            "pk": pub,
            "sk": keys["sk"],
            "params": keys["params"],
        },
    )


mcelc = mceliece_encrypt


def cheatsheet() -> str:
    return "mceliece_encrypt({}) -> Classic McEliece syndrome-based encryption."
