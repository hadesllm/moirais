# morie.fn — function file (hadesllm/morie)
"""Extract sorted eigenvalues. 'Rasengan!' -- Naruto, Naruto"""

from __future__ import annotations

from ._containers import DescriptiveResult


def extract_eigenvalues(B):
    """Return eigenvalues of symmetric matrix B sorted descending.

    Parameters
    ----------
    B : array-like
        Symmetric matrix.

    Returns
    -------
    DescriptiveResult
        value = sorted eigenvalues (ndarray).
    """
    import numpy as np

    B = np.asarray(B, dtype=float)
    vals = np.linalg.eigvalsh(B)
    vals = np.sort(vals)[::-1]
    return DescriptiveResult(name="extract_eigenvalues", value=vals, extra={"n": len(vals)})


eigvl = extract_eigenvalues


def cheatsheet() -> str:
    return "extract_eigenvalues({}) -> Extract sorted eigenvalues. 'Rasengan!' -- Naruto, Naruto"
