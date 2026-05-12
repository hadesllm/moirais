"""SHA-keyed deterministic RNG for Py<->R parity verification builds.

Given ``(name: str, seed: int)``, produces a :class:`numpy.random.Generator`
whose output is bit-identical to the corresponding R-side generator created
with :func:`morie_det_rng` for the same ``(name, seed)`` pair on the
canonical bootstrap / MCMC fixture.

Mechanism
---------
SHA-256(``name || ":" || seed``) is truncated to 32 bytes:

* the first 8 bytes seed a Philox(64-bit) BitGenerator on the Python side
  (drives ``numpy.random.Generator`` for permutations, normal draws, etc.);
* bytes ``[8:12]`` are emitted as a 31-bit signed integer that R passes to
  ``set.seed()`` so the base-R RNG state matches the Python counterpart.

The 64-bit Philox key + 31-bit R-side seed are independently derived from
disjoint slices of the SHA digest, so cross-language streams use the same
*statistical seed* without sharing an RNG implementation.  The contract is:

    from_seed(name, seed)              # Py: numpy Generator
    morie_det_rng(name, seed)          # R: integer passed to set.seed()

For verification builds the bootstrap or MCMC draws should agree to
Monte-Carlo tolerance (~2e-2 at B=2000 in v0.2.1) and bit-identical at
B->infinity if a deterministic-pseudo-bootstrap mode is plumbed.

This module is intentionally dependency-light: it imports only the
standard library + numpy, and the matching R helper uses base R `digest`
emulation via ``utils::sha256`` (R >= 4.3 ships ``tools::md5sum``;
the helper falls back to the ``digest`` / ``openssl`` packages when
present).
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["from_seed", "r_seed", "sha_digest_hex"]


def sha_digest_hex(name: str, seed: int) -> str:
    """Return SHA-256 hex digest of ``"{name}:{seed}"``.

    Exposed so R-side tests can cross-check identical hex digests.
    """
    return hashlib.sha256(f"{name}:{seed}".encode("utf-8")).hexdigest()


def from_seed(name: str, seed: int) -> np.random.Generator:
    """Build a numpy Generator from a SHA-keyed ``(name, seed)`` pair.

    Parameters
    ----------
    name : str
        Stable callable / fixture name, e.g. ``"ksr07_bootstrap"``.
    seed : int
        User-supplied integer seed.

    Returns
    -------
    numpy.random.Generator
        Philox-backed Generator whose stream is fully determined by
        SHA-256(name || ':' || seed).
    """
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    py_key = int.from_bytes(digest[:8], "big")  # 64-bit unsigned key
    bg = np.random.Philox(py_key)
    return np.random.Generator(bg)


def r_seed(name: str, seed: int) -> int:
    """Return the integer seed an R session should pass to ``set.seed()``.

    R's ``set.seed`` accepts a 32-bit signed integer; we emit a positive
    31-bit value derived from bytes ``[8:12]`` of the SHA digest so it is
    always representable.

    Parameters
    ----------
    name : str
        Stable callable / fixture name (same string the Py side uses).
    seed : int
        User-supplied integer seed.

    Returns
    -------
    int
        Positive integer in ``[0, 2**31 - 1)`` suitable for ``set.seed``.
    """
    digest = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[8:12], "big") % (2**31 - 1)
