# moirais.fn — function file (hadesllm/moirais)
"""JADE ICA via joint approximate diagonalization of eigenmatrices."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "We are what they grow beyond."


def jade_ica(X, n_components: int | None = None, max_iter: int = 100, tol: float = 1e-6, **kwargs) -> DescriptiveResult:
    """JADE ICA: joint approximate diagonalization of eigenmatrices.

    Separates sources by jointly diagonalising a set of cumulant
    matrices via Givens rotations.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data (rows are observations).
    n_components : int or None
        Number of sources. Default: n_features.
    max_iter : int
        Maximum sweeps of pairwise rotations (default 100).
    tol : float
        Convergence tolerance (default 1e-6).

    Returns
    -------
    DescriptiveResult
        ``value`` is n_components; ``extra`` has ``sources``,
        ``mixing_matrix``, ``unmixing_matrix``.

    References
    ----------
    Cardoso, J.-F., & Souloumiac, A. (1993). Blind beamforming for
    non-Gaussian signals. *IEE Proc. F*, 140(6), 362-370.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n_components is None:
        n_components = p
    n_components = min(n_components, p)

    mean = X.mean(axis=0)
    Xc = X - mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    K = (Vt[:n_components].T * (1.0 / (s[:n_components] + 1e-10))).T
    Z = (K @ Xc.T).T
    m = n_components

    cumulant_matrices = []
    for i in range(m):
        for j in range(i, m):
            M = np.zeros((m, m))
            for k in range(m):
                for l in range(m):
                    M[k, l] = np.mean(Z[:, i] * Z[:, j] * Z[:, k] * Z[:, l])
                    if i == j:
                        M[k, l] -= 1.0 if k == l else 0.0
                    if i == k and j == l:
                        M[k, l] -= 1.0
                    if i == l and j == k:
                        M[k, l] -= 1.0
            cumulant_matrices.append(M)

    V = np.eye(m)
    for _ in range(max_iter):
        total_off = 0.0
        for p_idx in range(m):
            for q_idx in range(p_idx + 1, m):
                a, b, c = 0.0, 0.0, 0.0
                for M in cumulant_matrices:
                    ip = M[p_idx, p_idx] - M[q_idx, q_idx]
                    iq = M[p_idx, q_idx] + M[q_idx, p_idx]
                    a += ip * ip
                    b += iq * iq
                    c += ip * iq
                theta = 0.5 * np.arctan2(2.0 * c, a - b + 1e-15)
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                total_off += abs(sin_t)
                G = np.eye(m)
                G[p_idx, p_idx] = cos_t
                G[q_idx, q_idx] = cos_t
                G[p_idx, q_idx] = -sin_t
                G[q_idx, p_idx] = sin_t
                for idx_m in range(len(cumulant_matrices)):
                    cumulant_matrices[idx_m] = G.T @ cumulant_matrices[idx_m] @ G
                V = V @ G
        if total_off < tol:
            break

    unmixing = V.T @ K
    sources = Xc @ unmixing.T
    mixing = np.linalg.pinv(unmixing)
    return DescriptiveResult(
        name="jade_ica",
        value=n_components,
        extra={"sources": sources, "mixing_matrix": mixing, "unmixing_matrix": unmixing},
    )


jade = jade_ica


def cheatsheet() -> str:
    return "jade_ica({}) -> JADE ICA via joint approximate diagonalization of eigenmatri"
