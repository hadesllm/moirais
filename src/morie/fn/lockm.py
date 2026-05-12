# morie.fn -- function file (hadesllm/morie)
"""Cryptographic hash (SHA-256). 'Some doors are meant to stay closed.' -- Lock"""

from __future__ import annotations

import hashlib

from ._containers import DescriptiveResult


def sha256_hash(
    message: str | bytes,
    *,
    encoding: str = "utf-8",
    iterations: int = 1,
) -> DescriptiveResult:
    """Compute SHA-256 hash of a message with optional iterated hashing.

    Parameters
    ----------
    message : str or bytes
        Input to hash.
    encoding : str
        Text encoding (ignored when *message* is bytes).
    iterations : int
        Number of times to re-hash the output (key stretching).
        Must be >= 1.

    Returns
    -------
    DescriptiveResult
        ``value`` is the hex digest string.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    if isinstance(message, str):
        data = message.encode(encoding)
    else:
        data = message

    digest = hashlib.sha256(data).digest()
    for _ in range(iterations - 1):
        digest = hashlib.sha256(digest).digest()

    hex_digest = digest.hex()
    return DescriptiveResult(
        name="SHA-256",
        value=hex_digest,
        extra={
            "length_bits": 256,
            "iterations": iterations,
            "input_bytes": len(data),
        },
    )


lockm = sha256_hash


def cheatsheet() -> str:
    return "sha256_hash({}) -> Cryptographic hash (SHA-256). 'Some doors are meant to stay "
