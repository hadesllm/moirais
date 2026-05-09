# moirais.fn — function file (hadesllm/moirais)
"""LiNGAM: Linear Non-Gaussian Acyclic Model for causal discovery.

LiNGAM identifies a unique causal ordering (not just an equivalence
class) by exploiting non-Gaussianity of the structural noise terms.

References
----------
Shimizu, S., Hoyer, P. O., Hyvarinen, A., & Kerminen, A. (2006).
A linear non-Gaussian acyclic model for causal discovery.
*Journal of Machine Learning Research*, 7, 2003-2030.

Hyvarinen, A., Zhang, K., Shimizu, S., & Hoyer, P. O. (2010).
Estimation of a structural vector autoregression model using non-
Gaussianity. *Journal of Machine Learning Research*, 11, 1709-1731.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["lingm"]


def lingm(
    X: np.ndarray,
    *,
    max_iter: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    r"""Estimate the LiNGAM causal order using ICA-based approach.

    The structural equation model is:

    .. math::

        \mathbf{x} = B \mathbf{x} + \mathbf{e}

    where :math:`B` is strictly lower-triangular (after causal
    permutation) and :math:`\mathbf{e}` has independent non-Gaussian
    components.

    This implementation uses FastICA to unmix the data, then
    identifies the causal permutation via minimum absolute-deviation
    matching on the W matrix (Shimizu et al. 2006, Algorithm 1).

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape ``(n, p)``.
    max_iter : int
        Maximum ICA iterations.
    seed : int
        Random seed for ICA initialisation.

    Returns
    -------
    dict
        ``B`` (estimated mixing matrix), ``order`` (causal order,
        index of root → leaf), ``W`` (unmixing matrix),
        ``method``, ``p``, ``n``.

    References
    ----------
    Shimizu et al. (2006). JMLR, 7, 2003-2030.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n, p).")
    n, p = X.shape

    # Centre
    X_c = X - X.mean(axis=0)

    # Whitening
    cov = X_c.T @ X_c / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    W_white = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
    X_white = X_c @ W_white.T

    # FastICA to estimate W_ica (unmixing)
    W_ica = _fastica(X_white, p, max_iter=max_iter, seed=seed)

    # Full unmixing matrix
    W = W_ica @ W_white

    # Permute rows of W to get closest to lower-triangular
    order = _find_causal_order(W)

    # Estimate B from W after permutation
    W_perm = W[order, :]
    W_perm = W_perm[:, order]
    # Normalize rows
    D_inv = np.diag(1.0 / np.diag(W_perm))
    B_hat = np.eye(p) - D_inv @ W_perm

    return {
        "B": B_hat,
        "order": order.tolist(),
        "W": W,
        "method": "LiNGAM",
        "p": p,
        "n": n,
    }


def _fastica(X: np.ndarray, n_comp: int, max_iter: int, seed: int) -> np.ndarray:
    """Minimal FastICA implementation using logcosh nonlinearity."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    W = rng.standard_normal((n_comp, p))
    W, _ = np.linalg.qr(W.T)
    W = W.T[:n_comp]

    for _ in range(max_iter):
        WX = W @ X.T                          # (n_comp, n)
        g = np.tanh(WX)                       # logcosh derivative
        gp = 1.0 - g**2                       # second derivative
        W_new = (g @ X) / n - gp.mean(axis=1, keepdims=True) * W
        # Symmetric orthogonalisation
        U, s, Vt = np.linalg.svd(W_new)
        W_new = U @ Vt
        if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0)) < 1e-6:
            W = W_new
            break
        W = W_new
    return W


def _find_causal_order(W: np.ndarray) -> np.ndarray:
    """Find permutation that makes W closest to lower-triangular.

    Iteratively removes the row with maximum absolute diagonal ratio
    (indicating weakest off-diagonal influence) per Shimizu et al.
    """
    p = W.shape[0]
    remaining = list(range(p))
    order = []
    Wc = W.copy()
    for _ in range(p):
        # Row with minimum absolute off-diagonal relative to diagonal
        scores = []
        for idx in remaining:
            diag_val = abs(Wc[idx, idx])
            if diag_val < 1e-12:
                scores.append(np.inf)
            else:
                off = [abs(Wc[idx, j]) for j in remaining if j != idx]
                scores.append(max(off) / diag_val if off else 0.0)
        best = remaining[int(np.argmin(scores))]
        order.append(best)
        remaining.remove(best)
    return np.array(order[::-1])


def cheatsheet() -> str:
    return "lingm(X) -> LiNGAM causal order via ICA (Shimizu et al. 2006, JMLR)."
