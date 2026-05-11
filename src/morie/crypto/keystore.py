"""Encrypted keystore for ML-KEM key pairs.

Stores ML-KEM-768 key pairs at ``~/.morie/keys/keystore.json``, encrypted
with a password-derived key (scrypt + ChaCha20-Poly1305).

WARNING: Research/educational implementation. NOT constant-time.
For production secrets, use OS-level keychains or hardware KMS.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from morie.crypto._chacha import chacha20_poly1305_decrypt, chacha20_poly1305_encrypt

_DEFAULT_PATH = "~/.morie/keys/keystore.json"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32


def _derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )


def _resolve(path: str) -> Path:
    return Path(path).expanduser()


def _read_store(path: str) -> dict:
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"Keystore not found: {p}")
    with open(p) as f:
        return json.load(f)


def _write_store(data: dict, path: str) -> None:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def create_keystore(password: str, path: str = _DEFAULT_PATH) -> None:
    """Create a new empty keystore.

    :param password: Password to protect the keystore.
    :param path: File path (default ``~/.morie/keys/keystore.json``).
    :raises FileExistsError: If the keystore already exists.
    """
    p = _resolve(path)
    if p.exists():
        raise FileExistsError(f"Keystore already exists: {p}")
    salt = os.urandom(16)
    _derive_key(password, salt)
    store = {"salt": salt.hex(), "keys": {}}
    _write_store(store, path)


def store_keypair(
    name: str,
    pk: bytes,
    sk: bytes,
    password: str,
    path: str = _DEFAULT_PATH,
) -> None:
    """Store a key pair in the keystore.

    The public key is stored in the clear. The secret key is encrypted
    with ChaCha20-Poly1305 using a password-derived key.

    :param name: Identifier for this key pair.
    :param pk: Public key bytes.
    :param sk: Secret key bytes.
    :param password: Keystore password.
    :param path: Keystore path.
    :raises FileNotFoundError: If the keystore does not exist.
    """
    store = _read_store(path)
    salt = bytes.fromhex(store["salt"])
    enc_key = _derive_key(password, salt)
    nonce = os.urandom(12)
    ct, tag = chacha20_poly1305_encrypt(enc_key, nonce, sk)
    store["keys"][name] = {
        "pk": pk.hex(),
        "sk_nonce": nonce.hex(),
        "sk_ct": ct.hex(),
        "sk_tag": tag.hex(),
    }
    _write_store(store, path)


def load_keypair(
    name: str,
    password: str,
    path: str = _DEFAULT_PATH,
) -> tuple[bytes, bytes]:
    """Load a key pair from the keystore.

    :param name: Identifier of the key pair.
    :param password: Keystore password.
    :param path: Keystore path.
    :return: ``(public_key, secret_key)`` as bytes.
    :raises KeyError: If the name is not found.
    :raises ValueError: If the password is wrong (tag mismatch).
    """
    store = _read_store(path)
    if name not in store["keys"]:
        raise KeyError(f"Key '{name}' not found in keystore")
    salt = bytes.fromhex(store["salt"])
    enc_key = _derive_key(password, salt)
    entry = store["keys"][name]
    nonce = bytes.fromhex(entry["sk_nonce"])
    ct = bytes.fromhex(entry["sk_ct"])
    tag = bytes.fromhex(entry["sk_tag"])
    sk = chacha20_poly1305_decrypt(enc_key, nonce, ct, tag)
    pk = bytes.fromhex(entry["pk"])
    return pk, sk


def list_keys(
    password: str,
    path: str = _DEFAULT_PATH,
) -> list[str]:
    """List key names in the keystore.

    Verifies the password by attempting to derive the key (the store
    must exist and the salt must be valid).

    :param password: Keystore password.
    :param path: Keystore path.
    :return: List of key names.
    """
    store = _read_store(path)
    salt = bytes.fromhex(store["salt"])
    _derive_key(password, salt)
    return list(store["keys"].keys())
