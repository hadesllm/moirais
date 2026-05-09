# moirais.fn — function file (hadesllm/moirais)
"""Scale coordinates Z = V * sqrt(Lambda). 'Gomu Gomu no...' -- Luffy, One Piece"""

from __future__ import annotations

from ._containers import DescriptiveResult


def scale_coordinates(eigvecs, eigvals):
    """Compute MDS coordinates: Z = V * sqrt(Lambda).

    Parameters
    ----------
    eigvecs : array-like
        Eigenvector matrix (n x p).
    eigvals : array-like
        Corresponding eigenvalues (length p).

    Returns
    -------
    DescriptiveResult
        value = coordinate matrix Z (n x p).
    """
    import numpy as np

    V = np.asarray(eigvecs, dtype=float)
    L = np.asarray(eigvals, dtype=float)
    L_pos = np.maximum(L, 0.0)
    Z = V * np.sqrt(L_pos)
    return DescriptiveResult(name="scale_coordinates", value=Z, extra={"n_dims": Z.shape[1]})


crdsc = scale_coordinates


def cheatsheet() -> str:
    return "scale_coordinates({}) -> Scale coordinates Z = V * sqrt(Lambda). 'Gomu Gomu no...' --"
