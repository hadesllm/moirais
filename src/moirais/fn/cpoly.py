# moirais.fn — function file (hadesllm/moirais)
"""ChaCha20-Poly1305 authenticated encryption (fn/ wrapper)."""

from __future__ import annotations

from ._containers import CryptoResult


def chacha20_encrypt(
    key: bytes,
    nonce: bytes,
    plaintext: bytes,
    aad: bytes = b"",
) -> CryptoResult:
    """ChaCha20-Poly1305 AEAD encryption.

    :param key: 32-byte key.
    :param nonce: 12-byte nonce.
    :param plaintext: Data to encrypt.
    :param aad: Additional authenticated data.
    :return: CryptoResult with ciphertext and tag in ``extra``.
    """
    from moirais.crypto._chacha import chacha20_poly1305_encrypt

    ct, tag = chacha20_poly1305_encrypt(key, nonce, plaintext, aad)
    return CryptoResult(
        algorithm="ChaCha20-Poly1305",
        operation="encrypt",
        success=True,
        extra={"ciphertext": ct, "tag": tag, "ct_len": len(ct)},
    )


def chacha20_decrypt(
    key: bytes,
    nonce: bytes,
    ciphertext: bytes,
    tag: bytes,
    aad: bytes = b"",
) -> CryptoResult:
    """ChaCha20-Poly1305 AEAD decryption.

    :param key: 32-byte key.
    :param nonce: 12-byte nonce.
    :param ciphertext: Data to decrypt.
    :param tag: 16-byte authentication tag.
    :param aad: Additional authenticated data.
    :return: CryptoResult with plaintext in ``extra``.
    :raises ValueError: If authentication fails.
    """
    from moirais.crypto._chacha import chacha20_poly1305_decrypt

    pt = chacha20_poly1305_decrypt(key, nonce, ciphertext, tag, aad)
    return CryptoResult(
        algorithm="ChaCha20-Poly1305",
        operation="decrypt",
        success=True,
        extra={"plaintext": pt},
    )


cpoly = chacha20_encrypt


def cheatsheet() -> str:
    return "chacha20_encrypt({}) -> ChaCha20-Poly1305 authenticated encryption (fn/ wrapper)."
