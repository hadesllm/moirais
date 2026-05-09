"""SOBI blind source separation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult

_QUOTE = "Hope is like the sun. If you only believe in it when you can see it."


def sobi_bss(X, n_sources: int | None = None, n_lags: int = 10, **kwargs) -> DescriptiveResult:
    """Second-Order Blind Identification for source separation.

    Uses time-lagged covariance matrices and joint diagonalisation
    to separate temporally correlated sources.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_channels)
        Mixed signal matrix.
    n_sources : int or None
        Number of sources. Default: n_channels.
    n_lags : int
        Number of time lags for covariance estimation (default 10).

    Returns
    -------
    DescriptiveResult
        ``value`` is n_sources; ``extra`` has ``sources``,
        ``mixing_matrix``, ``unmixing_matrix``.

    References
    ----------
    Belouchrani, A., Abed-Meraim, K., Cardoso, J.-F., & Moulines, E.
    (1997). A blind source separation technique using second-order
    statistics. *IEEE Trans. Signal Process.*, 45(2), 434-444.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if n_sources is None:
        n_sources = p
    n_sources = min(n_sources, p)

    mean = X.mean(axis=0)
    Xc = X - mean
    cov0 = Xc.T @ Xc / n
    eigvals, eigvecs = np.linalg.eigh(cov0)
    idx = np.argsort(eigvals)[::-1][:n_sources]
    D = np.diag(1.0 / np.sqrt(eigvals[idx] + 1e-10))
    W = D @ eigvecs[:, idx].T
    Z = (W @ Xc.T).T

    cov_mats = []
    for lag in range(1, n_lags + 1):
        Rlag = Z[lag:].T @ Z[:-lag] / (n - lag)
        Rlag = (Rlag + Rlag.T) / 2.0
        cov_mats.append(Rlag)

    V = np.eye(n_sources)
    for _ in range(100):
        total_off = 0.0
        for pi in range(n_sources):
            for qi in range(pi + 1, n_sources):
                a, b, c = 0.0, 0.0, 0.0
                for R in cov_mats:
                    h = R[pi, pi] - R[qi, qi]
                    g = R[pi, qi] + R[qi, pi]
                    a += h * h
                    b += g * g
                    c += h * g
                theta = 0.5 * np.arctan2(2.0 * c, a - b + 1e-15)
                cs = np.cos(theta)
                sn = np.sin(theta)
                total_off += abs(sn)
                G = np.eye(n_sources)
                G[pi, pi] = cs
                G[qi, qi] = cs
                G[pi, qi] = -sn
                G[qi, pi] = sn
                for k in range(len(cov_mats)):
                    cov_mats[k] = G.T @ cov_mats[k] @ G
                V = V @ G
        if total_off < 1e-8:
            break

    unmixing = V.T @ W
    sources = Xc @ unmixing.T
    mixing = np.linalg.pinv(unmixing)
    return DescriptiveResult(
        name="sobi_bss",
        value=n_sources,
        extra={"sources": sources, "mixing_matrix": mixing, "unmixing_matrix": unmixing},
    )


sobi = sobi_bss


def cheatsheet() -> str:
    return "sobi_bss({}) -> SOBI blind source separation."
