# moirais.fn — function file (hadesllm/moirais)
"""Matching pursuit decomposition."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "The belonging you seek is not behind you, it is ahead."


def matching_pursuit_decompose(x, dictionary=None, sparsity: int = 10, **kwargs) -> DescriptiveResult:
    """Matching pursuit decomposition.

    Parameters
    ----------
    x : array-like
        Input signal (1-D).
    dictionary : array-like or None
        Dictionary matrix (n_features x n_atoms). If None, uses a DCT
        dictionary of size 2*len(x).
    sparsity : int
        Number of atoms to select (default 10).

    Returns
    -------
    DescriptiveResult
        ``value`` is residual norm; ``extra`` has ``coefficients``,
        ``atoms_used``, ``residual``, ``approximation``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if dictionary is None:
        n_atoms = 2 * n
        D = np.zeros((n, n_atoms))
        for k in range(n_atoms):
            for i in range(n):
                D[i, k] = np.cos(np.pi * (i + 0.5) * k / n_atoms)
            norm = np.linalg.norm(D[:, k])
            if norm > 1e-15:
                D[:, k] /= norm
    else:
        D = np.asarray(dictionary, dtype=float)

    residual = x.copy()
    coefficients = np.zeros(D.shape[1])
    atoms_used = []

    for _ in range(sparsity):
        corrs = np.abs(D.T @ residual)
        best = int(np.argmax(corrs))
        atoms_used.append(best)
        coeff = D[:, best] @ residual
        coefficients[best] += coeff
        residual = residual - coeff * D[:, best]

    approx = x - residual
    res_norm = float(np.linalg.norm(residual))

    return DescriptiveResult(
        name="matching_pursuit_decompose",
        value=res_norm,
        extra={"coefficients": coefficients, "atoms_used": atoms_used, "residual": residual, "approximation": approx},
    )


mpdcm = matching_pursuit_decompose


def cheatsheet() -> str:
    return "matching_pursuit_decompose({}) -> Matching pursuit decomposition."
