"""ChaCha20-Poly1305 AEAD (RFC 8439).

Pure-Python reference implementation. NOT constant-time.
For production use, prefer libsodium or OS-level crypto.
"""

from __future__ import annotations

import struct


def _rotl32(v: int, n: int) -> int:
    return ((v << n) | (v >> (32 - n))) & 0xFFFFFFFF


def _quarter_round(state: list[int], a: int, b: int, c: int, d: int) -> None:
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = _rotl32(state[d], 16)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = _rotl32(state[b], 12)
    state[a] = (state[a] + state[b]) & 0xFFFFFFFF
    state[d] ^= state[a]
    state[d] = _rotl32(state[d], 8)
    state[c] = (state[c] + state[d]) & 0xFFFFFFFF
    state[b] ^= state[c]
    state[b] = _rotl32(state[b], 7)


def _chacha20_block(key: bytes, counter: int, nonce: bytes) -> bytes:
    state = list(struct.unpack("<16I", b"expand 32-byte k" + key + struct.pack("<I", counter) + nonce))
    working = state[:]
    for _ in range(10):
        _quarter_round(working, 0, 4, 8, 12)
        _quarter_round(working, 1, 5, 9, 13)
        _quarter_round(working, 2, 6, 10, 14)
        _quarter_round(working, 3, 7, 11, 15)
        _quarter_round(working, 0, 5, 10, 15)
        _quarter_round(working, 1, 6, 11, 12)
        _quarter_round(working, 2, 7, 8, 13)
        _quarter_round(working, 3, 4, 9, 14)
    out = struct.pack("<16I", *[(working[i] + state[i]) & 0xFFFFFFFF for i in range(16)])
    return out


def _chacha20_encrypt_bytes(key: bytes, counter: int, nonce: bytes, data: bytes) -> bytes:
    result = bytearray()
    for i in range(0, len(data), 64):
        block = _chacha20_block(key, counter + i // 64, nonce)
        chunk = data[i : i + 64]
        result.extend(bytes(a ^ b for a, b in zip(chunk, block[: len(chunk)])))
    return bytes(result)


def _clamp(r: int) -> int:
    return r & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF


def _poly1305_mac(key: bytes, msg: bytes) -> bytes:
    r = _clamp(int.from_bytes(key[:16], "little"))
    s = int.from_bytes(key[16:32], "little")
    p = (1 << 130) - 5
    acc = 0
    for i in range(0, len(msg), 16):
        chunk = msg[i : i + 16]
        n = int.from_bytes(chunk, "little") + (1 << (8 * len(chunk)))
        acc = ((acc + n) * r) % p
    acc = (acc + s) & ((1 << 128) - 1)
    return acc.to_bytes(16, "little")


def _pad16(data: bytes) -> bytes:
    rem = len(data) % 16
    if rem == 0:
        return b""
    return b"\x00" * (16 - rem)


def chacha20_poly1305_encrypt(
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    aad: bytes = b"",
) -> tuple[bytes, bytes]:
    """ChaCha20-Poly1305 AEAD encryption (RFC 8439).

    :param key: 32-byte key.
    :param nonce: 12-byte nonce.
    :param plaintext: Data to encrypt.
    :param aad: Additional authenticated data.
    :return: (ciphertext, 16-byte tag).
    :raises ValueError: If key or nonce has wrong length.
    """
    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes, got {len(key)}")
    if len(nonce) != 12:
        raise ValueError(f"Nonce must be 12 bytes, got {len(nonce)}")

    otk = _chacha20_block(key, 0, nonce)[:32]
    ciphertext = _chacha20_encrypt_bytes(key, 1, nonce, plaintext)

    mac_data = (
        aad
        + _pad16(aad)
        + ciphertext
        + _pad16(ciphertext)
        + struct.pack("<Q", len(aad))
        + struct.pack("<Q", len(ciphertext))
    )
    tag = _poly1305_mac(otk, mac_data)
    return ciphertext, tag


def chacha20_poly1305_decrypt(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    aad: bytes = b"",
) -> bytes:
    """ChaCha20-Poly1305 AEAD decryption (RFC 8439).

    :param key: 32-byte key.
    :param nonce: 12-byte nonce.
    :param ciphertext: Data to decrypt.
    :param tag: 16-byte authentication tag.
    :param aad: Additional authenticated data.
    :return: Decrypted plaintext.
    :raises ValueError: If authentication fails or parameters invalid.
    """
    if len(key) != 32:
        raise ValueError(f"Key must be 32 bytes, got {len(key)}")
    if len(nonce) != 12:
        raise ValueError(f"Nonce must be 12 bytes, got {len(nonce)}")

    otk = _chacha20_block(key, 0, nonce)[:32]
    mac_data = (
        aad
        + _pad16(aad)
        + ciphertext
        + _pad16(ciphertext)
        + struct.pack("<Q", len(aad))
        + struct.pack("<Q", len(ciphertext))
    )
    expected_tag = _poly1305_mac(otk, mac_data)

    if expected_tag != tag:
        raise ValueError("Authentication failed: tag mismatch")

    return _chacha20_encrypt_bytes(key, 1, nonce, ciphertext)
