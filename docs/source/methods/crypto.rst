Post-Quantum Cryptography (Research)
=====================================

Part of :doc:`index` — MORIE's statistical-methods reference.

.. warning::

   MORIE crypto is a **teaching/research implementation** of NIST-standardized
   post-quantum cryptography. For production-grade secrets, use hardware KMS
   or libsodium. Pure-Python ML-KEM is ~100x slower than the reference C
   implementation and is **not constant-time** (vulnerable to timing attacks).

MORIE provides a pure-Python post-quantum cryptography module with no external
dependencies, suitable for educational use and non-critical research data.

ML-KEM-768 (FIPS 203)
----------------------

Module Lattice-based Key Encapsulation Mechanism, formerly Kyber768.
Provides IND-CCA2-secure key encapsulation using the module-LWE problem.

.. code-block:: python

   from morie.crypto import mlkem768_keygen, mlkem768_encaps, mlkem768_decaps

   pk, sk = mlkem768_keygen()
   ct, shared_secret = mlkem768_encaps(pk)
   recovered = mlkem768_decaps(sk, ct)

Parameters: q=3329, n=256, k=3, eta1=2, eta2=2.

ChaCha20-Poly1305 (RFC 8439)
-----------------------------

Authenticated encryption with associated data (AEAD). Provides
confidentiality and integrity for arbitrary-length messages.

.. code-block:: python

   from morie.crypto import chacha20_poly1305_encrypt, chacha20_poly1305_decrypt

   ct = chacha20_poly1305_encrypt(key, nonce, plaintext, aad=b"")
   pt = chacha20_poly1305_decrypt(key, nonce, ct, aad=b"")

HKDF-SHA256 (RFC 5869)
-----------------------

Key derivation function for expanding keying material.

.. code-block:: python

   from morie.crypto import hkdf_sha256
   derived = hkdf_sha256(input_key_material, length=32, salt=b"", info=b"")

Hybrid KEM-DEM Construction
----------------------------

The recommended usage combines ML-KEM for key encapsulation with
ChaCha20-Poly1305 for data encryption:

.. code-block:: python

   from morie.crypto import hybrid_encrypt, hybrid_decrypt, hybrid_keygen

   pk, sk = hybrid_keygen()
   ciphertext = hybrid_encrypt(b"secret data", pk)
   plaintext = hybrid_decrypt(ciphertext, sk)

Container format: ``kem_ct_len(4B) || kem_ct || nonce(12B) || aead_ct || tag(16B)``

CLI Usage
---------

.. code-block:: bash

   morie crypto keygen --name alice --output ./keys
   morie crypto encrypt secret.csv --to ./keys/alice_pk.bin --output secret.morieenc
   morie crypto decrypt secret.morieenc --sk ./keys/alice_sk.bin --output secret.csv

Keystore
--------

Encrypted keystore at ``~/.morie/keys/keystore.json`` with scrypt-derived
password protection:

.. code-block:: python

   from morie.crypto import create_keystore, store_keypair, load_keypair

   create_keystore("my-password")
   pk, sk = hybrid_keygen()
   store_keypair("alice", pk, sk, "my-password")
   pk2, sk2 = load_keypair("alice", "my-password")

Security Considerations
-----------------------

- **Not constant-time**: Vulnerable to timing side-channels. Do not use
  for high-value secrets in adversarial environments.
- **No hardware acceleration**: Pure Python; ~100x slower than libsodium/C.
- **Research-grade**: Suitable for coursework, CTF challenges, and
  non-critical research data protection.
- **Key material**: Never commit ``*.moriesk``, ``*.morieenc``, or
  ``~/.morie/keys/`` to version control.
