"""Wavelet dictionary learning."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "I find your lack of faith disturbing."


def wavelet_dict_learn(
    X, n_atoms: int = 20, wavelet: str = "db4", n_iter: int = 30, sparsity: int = 3, **kwargs
) -> DescriptiveResult:
    """Wavelet-initialized dictionary learning.

    Initializes dictionary atoms using Haar-like wavelet patterns,
    then refines via K-SVD-style updates.

    Parameters
    ----------
    X : array-like, shape (n_features, n_samples)
        Data matrix (each column is a sample).
    n_atoms : int
        Number of dictionary atoms (default 20).
    wavelet : str
        Wavelet type for initialization (default "db4"). Currently uses
        Haar-like patterns regardless of name.
    n_iter : int
        Number of update iterations (default 30).
    sparsity : int
        Max non-zero coefficients per sample (default 3).

    Returns
    -------
    DescriptiveResult
        ``value`` is final representation error; ``extra`` has ``dictionary``,
        ``coefficients``, ``wavelet``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n_features, n_samples = X.shape

    D = np.zeros((n_features, n_atoms))
    rng = np.random.default_rng(0)
    for j in range(n_atoms):
        scale = max(1, n_features // (j + 1))
        atom = np.zeros(n_features)
        start = rng.integers(0, max(1, n_features - scale))
        half = scale // 2
        atom[start : start + half] = 1.0
        atom[start + half : start + scale] = -1.0
        norm = np.linalg.norm(atom)
        if norm > 1e-15:
            atom /= norm
        else:
            atom = rng.standard_normal(n_features)
            atom /= np.linalg.norm(atom)
        D[:, j] = atom

    C = np.zeros((n_atoms, n_samples))

    for _ in range(n_iter):
        for i in range(n_samples):
            residual = X[:, i].copy()
            idx = []
            for _ in range(sparsity):
                corrs = np.abs(D.T @ residual)
                corrs[idx] = -1
                best = int(np.argmax(corrs))
                idx.append(best)
                Ds = D[:, idx]
                coeffs, _, _, _ = np.linalg.lstsq(Ds, X[:, i], rcond=None)
                residual = X[:, i] - Ds @ coeffs
            C[:, i] = 0.0
            C[idx, i] = coeffs

        for j in range(n_atoms):
            mask = np.abs(C[j, :]) > 1e-12
            if not np.any(mask):
                continue
            Ej = X[:, mask] - D @ C[:, mask] + np.outer(D[:, j], C[j, mask])
            U, s, Vt = np.linalg.svd(Ej, full_matrices=False)
            D[:, j] = U[:, 0]
            C[j, mask] = s[0] * Vt[0, :]

    err = np.linalg.norm(X - D @ C) / (np.linalg.norm(X) + 1e-15)
    return DescriptiveResult(
        name="wavelet_dict_learn",
        value=float(err),
        extra={"dictionary": D, "coefficients": C, "wavelet": wavelet},
    )


wvdl = wavelet_dict_learn


def cheatsheet() -> str:
    return "wavelet_dict_learn({}) -> Wavelet dictionary learning."
