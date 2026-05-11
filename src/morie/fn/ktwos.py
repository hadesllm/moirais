# morie.fn — function file (hadesllm/morie)
"""Kernel two-sample test (Maximum Mean Discrepancy)."""

from __future__ import annotations

import numpy as np
from ._richresult import RichResult

__all__ = ["ktwos"]


def ktwos(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bw: float | None = None,
    n_perm: int = 999,
    seed: int = 42,
) -> dict:
    r"""
    Kernel two-sample test based on Maximum Mean Discrepancy (MMD).

    Tests :math:`H_0: F_X = F_Y` using the unbiased MMD^2 estimator
    with a Gaussian kernel:

    .. math::

        \widehat{\mathrm{MMD}}^2_u =
        \frac{1}{m(m-1)}\sum_{i\ne j}k(X_i,X_j)
        + \frac{1}{n(n-1)}\sum_{i\ne j}k(Y_i,Y_j)
        - \frac{2}{mn}\sum_{i,j}k(X_i,Y_j)

    Parameters
    ----------
    x : np.ndarray
        Sample 1 (1-d).
    y : np.ndarray
        Sample 2 (1-d).
    bw : float or None
        Gaussian kernel bandwidth. If None, uses the median heuristic.
    n_perm : int
        Permutation replicates for the p-value.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``mmd2``, ``p_value``, ``bw``.

    References
    ----------
    Gretton, A. et al. (2012). A kernel two-sample test. *JMLR*, 13,
        723-773.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m, nn = len(x), len(y)
    if m < 2 or nn < 2:
        raise ValueError("Both samples need at least 2 observations.")

    if bw is None:
        combined = np.concatenate([x, y])
        dists = np.abs(combined[:, None] - combined[None, :])
        bw = float(np.median(dists[dists > 0]))
        bw = max(bw, 1e-10)

    def _rbf(a, b, h):
        d = a[:, None] - b[None, :]
        return np.exp(-0.5 * (d / h) ** 2)

    def _mmd2(a, b, h):
        kxx = _rbf(a, a, h)
        np.fill_diagonal(kxx, 0.0)
        kyy = _rbf(b, b, h)
        np.fill_diagonal(kyy, 0.0)
        kxy = _rbf(a, b, h)
        la, lb = len(a), len(b)
        return (kxx.sum() / (la * (la - 1))
                + kyy.sum() / (lb * (lb - 1))
                - 2.0 * kxy.mean())

    stat = _mmd2(x, y, bw)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([x, y])
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(m + nn)
        perm_stats[i] = _mmd2(pooled[idx[:m]], pooled[idx[m:]], bw)

    p_value = float(np.mean(perm_stats >= stat))

    return RichResult(payload={"mmd2": float(stat), "p_value": p_value, "bw": bw})


def cheatsheet() -> str:
    return "ktwos(x, y) -> Kernel two-sample test (MMD)."
