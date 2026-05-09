# moirais.fn — function file (hadesllm/moirais)
"""K-SVD dictionary learning."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Statistics is the grammar of science. — Karl Pearson"


def ksvd_dictionary(Y, n_atoms: int = 20, sparsity: int = 3, n_iter: int = 50, **kwargs) -> DescriptiveResult:
    """K-SVD dictionary learning.

    Parameters
    ----------
    Y : array-like, shape (n_features, n_samples)
        Data matrix (each column is a sample).
    n_atoms : int
        Number of dictionary atoms (default 20).
    sparsity : int
        Max non-zero coefficients per sample (default 3).
    n_iter : int
        Number of K-SVD iterations (default 50).

    Returns
    -------
    DescriptiveResult
        ``value`` is final representation error; ``extra`` has ``dictionary``
        (n_features x n_atoms), ``coefficients`` (n_atoms x n_samples),
        ``iterations``.

    References
    ----------
    Aharon, M., Elad, M., & Bruckstein, A. (2006). K-SVD: An algorithm
    for designing overcomplete dictionaries. *IEEE TSP*, 54(11), 4311-4322.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_features, n_samples = Y.shape
    rng = np.random.default_rng(0)

    D = rng.standard_normal((n_features, n_atoms))
    for j in range(n_atoms):
        D[:, j] /= np.linalg.norm(D[:, j]) + 1e-15

    X = np.zeros((n_atoms, n_samples))

    def _omp(D, y, T):
        residual = y.copy()
        idx = []
        for _ in range(T):
            corrs = np.abs(D.T @ residual)
            corrs[idx] = -1
            best = int(np.argmax(corrs))
            idx.append(best)
            Ds = D[:, idx]
            coeffs, _, _, _ = np.linalg.lstsq(Ds, y, rcond=None)
            residual = y - Ds @ coeffs
        c = np.zeros(D.shape[1])
        c[idx] = coeffs
        return c

    for it in range(n_iter):
        for i in range(n_samples):
            X[:, i] = _omp(D, Y[:, i], sparsity)

        for j in range(n_atoms):
            mask = np.abs(X[j, :]) > 1e-12
            if not np.any(mask):
                continue
            Ej = Y[:, mask] - D @ X[:, mask] + np.outer(D[:, j], X[j, mask])
            U, s, Vt = np.linalg.svd(Ej, full_matrices=False)
            D[:, j] = U[:, 0]
            X[j, mask] = s[0] * Vt[0, :]

    err = np.linalg.norm(Y - D @ X) / (np.linalg.norm(Y) + 1e-15)
    return DescriptiveResult(
        name="ksvd_dictionary",
        value=float(err),
        extra={"dictionary": D, "coefficients": X, "iterations": n_iter},
    )


ksvd = ksvd_dictionary


def cheatsheet() -> str:
    return "ksvd_dictionary({}) -> K-SVD dictionary learning."
