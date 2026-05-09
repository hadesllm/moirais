# moirais.fn — function file (hadesllm/moirais)
"""Extract top eigenvectors. 'Gear Fifth!' -- Luffy, One Piece"""

from __future__ import annotations

from ._containers import DescriptiveResult


def extract_eigenvectors(B, n_dims=2):
    """Return top *n_dims* eigenvectors of symmetric matrix B.

    Parameters
    ----------
    B : array-like
        Symmetric matrix.
    n_dims : int
        Number of eigenvectors to extract.

    Returns
    -------
    DescriptiveResult
        value = eigenvector matrix (n x n_dims), extra has eigenvalues.
    """
    import numpy as np

    B = np.asarray(B, dtype=float)
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    return DescriptiveResult(
        name="extract_eigenvectors",
        value=vecs[:, :n_dims],
        extra={"eigenvalues": vals[:n_dims], "n_dims": n_dims},
    )


eigvc = extract_eigenvectors


def cheatsheet() -> str:
    return "extract_eigenvectors({}) -> Extract top eigenvectors. 'Gear Fifth!' -- Luffy, One Piece"
