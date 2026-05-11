"""HKDF-SHA256 key derivation (RFC 5869)."""

from __future__ import annotations

import hashlib
import hmac
import math


def hkdf_sha256(
    ikm: bytes,
    length: int = 32,
    salt: bytes = b"",
    info: bytes = b"",
) -> bytes:
    """HKDF-SHA256 key derivation function.

    :param ikm: Input keying material.
    :param length: Output length in bytes (max 8160).
    :param salt: Optional salt (if empty, uses zero-filled hash-length).
    :param info: Optional context/application info.
    :return: Derived key material.
    """
    hash_len = 32
    if length > 255 * hash_len:
        raise ValueError(f"Output length {length} exceeds maximum {255 * hash_len}")

    if not salt:
        salt = b"\x00" * hash_len

    prk = hmac.new(salt, ikm, hashlib.sha256).digest()

    n = math.ceil(length / hash_len)
    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t

    return okm[:length]
