# morie.fn -- function file (hadesllm/morie)
"""Word embedding lookup (Mikolov et al. 2013)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["word_embedding"]


def word_embedding(x, E=None, vocab_size: int = 100, d_model: int = 16,
                   seed: int = 0):
    """Look up word embeddings from a (V x d) matrix.

    Formula: ``e = E[token_id]``, where ``E`` has shape ``(vocab_size, d_model)``.

    Parameters
    ----------
    x : int or array-like of int
        Token id(s) to look up.
    E : ndarray of shape (V, d), optional
        Embedding matrix.  If None, a deterministic Glorot-uniform
        matrix of shape (vocab_size, d_model) is generated from ``seed``.
    vocab_size, d_model : int
        Used only when ``E`` is None.
    seed : int
        RNG seed for the synthetic embedding matrix.

    Returns
    -------
    RichResult with keys: tensor (e), E, ids, shape.
    """
    ids = np.atleast_1d(np.asarray(x, dtype=int))
    if E is None:
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (vocab_size + d_model))
        E = rng.uniform(-limit, limit, size=(vocab_size, d_model))
    E = np.asarray(E, dtype=float)
    if ids.max(initial=-1) >= E.shape[0] or ids.min(initial=0) < 0:
        raise IndexError("token id out of range for embedding matrix")
    e = E[ids]
    return RichResult(
        title="Word Embedding Lookup (Mikolov 2013)",
        summary_lines=[("ids", ids.tolist()),
                       ("d_model", E.shape[1])],
        payload={"tensor": e, "E": E, "ids": ids,
                 "shape": e.shape, "method": "embedding-lookup"},
    )


def cheatsheet():
    return "wdemb(ids, E): embedding-matrix row lookup"


# CANONICAL TEST
# >>> E = np.eye(4)
# >>> r = word_embedding([0, 2], E=E)
# >>> np.allclose(r["tensor"], np.array([[1,0,0,0],[0,0,1,0]]))
# True
