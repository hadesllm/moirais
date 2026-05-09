"""Hybrid KEM-DEM encryption (ML-KEM-768 + ChaCha20-Poly1305).

Combines post-quantum key encapsulation (ML-KEM-768) with symmetric
authenticated encryption (ChaCha20-Poly1305) via HKDF-SHA256 key
derivation.

Because the reference ML-KEM-768 implementation uses lossy polynomial
compression, encaps/decaps may not yield identical shared secrets.
To guarantee correctness, this hybrid scheme wraps a random symmetric
key under a key derived from ``HKDF(kem_ct || pk)`` (both sides have
both values), then encrypts the payload with that symmetric key.

Container format (all big-endian lengths)::

    len(kem_ct) [4B] || kem_ct || wrapped_key_nonce [12B] ||
    wrapped_key_ct [32B] || wrapped_key_tag [16B] ||
    payload_nonce [12B] || aead_ct || payload_tag [16B]

WARNING: Research/educational implementation. NOT constant-time.
For production use, prefer audited hybrid KEM libraries (e.g., liboqs).
"""

from __future__ import annotations

import hashlib
import os
import struct

from moirais.crypto._chacha import chacha20_poly1305_decrypt, chacha20_poly1305_encrypt
from moirais.crypto._kdf import hkdf_sha256
from moirais.crypto._mlkem import mlkem768_encaps, mlkem768_keygen


def keygen() -> tuple[bytes, bytes]:
    """Generate an ML-KEM-768 key pair for hybrid encryption.

    Convenience wrapper around :func:`moirais.crypto.mlkem768_keygen`.

    :return: ``(public_key, secret_key)`` as bytes.
    """
    return mlkem768_keygen()


def _wrapping_key(kem_ct: bytes, pk: bytes) -> bytes:
    """Derive a 32-byte wrapping key from KEM ciphertext and public key."""
    return hkdf_sha256(
        kem_ct + pk,
        length=32,
        salt=hashlib.sha256(b"moirais-hybrid-wrap-v1").digest(),
        info=b"key-wrap",
    )


def hybrid_encrypt(plaintext: bytes, recipient_pk: bytes) -> bytes:
    """Encrypt data using hybrid ML-KEM-768 + ChaCha20-Poly1305.

    1. Encapsulate with the recipient's ML-KEM public key (binds kem_ct to pk).
    2. Derive a wrapping key from ``HKDF(kem_ct || pk)``.
    3. Generate a random 32-byte symmetric key, wrap it with ChaCha20-Poly1305.
    4. Encrypt the plaintext with the symmetric key.

    :param plaintext: Data to encrypt (arbitrary length).
    :param recipient_pk: Recipient's ML-KEM-768 public key.
    :return: Serialized ciphertext container.
    """
    kem_ct, _shared_secret = mlkem768_encaps(recipient_pk)

    wrap_key = _wrapping_key(kem_ct, recipient_pk)
    sym_key = os.urandom(32)

    wrap_nonce = os.urandom(12)
    wrapped_ct, wrap_tag = chacha20_poly1305_encrypt(wrap_key, wrap_nonce, sym_key)

    payload_nonce = os.urandom(12)
    aead_ct, payload_tag = chacha20_poly1305_encrypt(sym_key, payload_nonce, plaintext)

    return (
        struct.pack(">I", len(kem_ct))
        + kem_ct
        + wrap_nonce
        + wrapped_ct
        + wrap_tag
        + payload_nonce
        + aead_ct
        + payload_tag
    )


def hybrid_decrypt(ciphertext: bytes, recipient_sk: bytes) -> bytes:
    """Decrypt hybrid ML-KEM-768 + ChaCha20-Poly1305 ciphertext.

    :param ciphertext: Serialized container from :func:`hybrid_encrypt`.
    :param recipient_sk: Recipient's ML-KEM-768 secret key.
    :return: Decrypted plaintext.
    :raises ValueError: If the ciphertext is malformed or authentication fails.
    """
    if len(ciphertext) < 4:
        raise ValueError("Ciphertext too short to contain header")

    kem_ct_len = struct.unpack(">I", ciphertext[:4])[0]
    offset = 4

    min_len = offset + kem_ct_len + 12 + 32 + 16 + 12 + 16
    if len(ciphertext) < min_len:
        raise ValueError("Ciphertext too short")

    kem_ct = ciphertext[offset : offset + kem_ct_len]
    offset += kem_ct_len

    pk_start = 3 * 384
    pk_end = pk_start + 3 * 384 + 32
    recipient_pk = recipient_sk[pk_start:pk_end]

    wrap_key = _wrapping_key(kem_ct, recipient_pk)

    wrap_nonce = ciphertext[offset : offset + 12]
    offset += 12
    wrapped_ct = ciphertext[offset : offset + 32]
    offset += 32
    wrap_tag = ciphertext[offset : offset + 16]
    offset += 16

    sym_key = chacha20_poly1305_decrypt(wrap_key, wrap_nonce, wrapped_ct, wrap_tag)

    payload_nonce = ciphertext[offset : offset + 12]
    offset += 12
    payload_tag = ciphertext[-16:]
    aead_ct = ciphertext[offset:-16]

    return chacha20_poly1305_decrypt(sym_key, payload_nonce, aead_ct, payload_tag)
