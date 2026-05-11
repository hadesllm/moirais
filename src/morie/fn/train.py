"""Domain adaptation via MMD. 'I can take you anywhere, but you gotta want to go.' -- The Trainman"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def mmd_distance(
    source: np.ndarray,
    target: np.ndarray,
    *,
    kernel: str = "rbf",
    bandwidth: float | None = None,
) -> DescriptiveResult:
    """Maximum Mean Discrepancy between two samples.

    Estimates the two-sample MMD^2 using a kernel trick, which measures
    the distance between the embeddings of two distributions in a
    reproducing kernel Hilbert space (RKHS).

    Parameters
    ----------
    source : ndarray
        Source domain samples (n_s x p).
    target : ndarray
        Target domain samples (n_t x p).
    kernel : str
        ``"rbf"`` (Gaussian) or ``"linear"``.
    bandwidth : float or None
        RBF kernel bandwidth (sigma). If None, uses the median heuristic.

    Returns
    -------
    DescriptiveResult
        ``value`` is the MMD^2 estimate; ``extra`` has bandwidth and
        per-term contributions.

    References
    ----------
    Gretton, A. et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
    """
    S = np.asarray(source, dtype=np.float64)
    T = np.asarray(target, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    if S.shape[1] != T.shape[1]:
        raise ValueError("source and target must have the same number of features")

    n_s, n_t = S.shape[0], T.shape[0]
    if n_s < 2 or n_t < 2:
        raise ValueError("Need at least 2 samples in each domain")

    def _pdist2(A, B):
        return np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)

    if kernel == "linear":
        k_ss = float(np.mean(S @ S.T))
        k_tt = float(np.mean(T @ T.T))
        k_st = float(np.mean(S @ T.T))
    elif kernel == "rbf":
        if bandwidth is None:
            all_pts = np.vstack([S, T])
            dists = _pdist2(all_pts, all_pts)
            bandwidth = float(np.median(dists[dists > 0]) ** 0.5)
            if bandwidth == 0:
                bandwidth = 1.0

        gamma = 1.0 / (2 * bandwidth**2)
        k_ss = float(np.mean(np.exp(-gamma * _pdist2(S, S))))
        k_tt = float(np.mean(np.exp(-gamma * _pdist2(T, T))))
        k_st = float(np.mean(np.exp(-gamma * _pdist2(S, T))))
    else:
        raise ValueError(f"kernel must be 'rbf' or 'linear', got '{kernel}'")

    mmd2 = k_ss + k_tt - 2 * k_st

    return DescriptiveResult(
        name="MMD Distance",
        value=float(max(mmd2, 0.0)),
        extra={
            "kernel": kernel,
            "bandwidth": bandwidth,
            "k_ss": k_ss,
            "k_tt": k_tt,
            "k_st": k_st,
            "n_source": n_s,
            "n_target": n_t,
        },
    )


train = mmd_distance


def cheatsheet() -> str:
    return "mmd_distance({}) -> Domain adaptation via MMD. 'I can take you anywhere, but you"
