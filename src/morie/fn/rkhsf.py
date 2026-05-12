# morie.fn — function file (hadesllm/morie)
"""RKHS regression with a Gaussian kernel for genomic prediction."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["rkhs_full"]


def rkhs_full(x, y, markers, h: float | None = None, lam: float = 1.0):
    """RKHS regression (kernel ridge) with Gaussian kernel.

    Model::

        y = X*beta + K*alpha + e,  K_ij = exp(-||m_i - m_j||^2 / h)

    Solved as a kernel-ridge / RKHS BLUP: alpha = (K + lam*I)^{-1} (y - X*beta).
    `h` defaults to the median squared Euclidean distance between marker rows
    (Jaakkola heuristic).

    Parameters
    ----------
    x : array-like (n,) or (n,p)
    y : array-like (n,)
    markers : array-like (n,m)
    h : float, optional. Kernel bandwidth. Default = median ||m_i - m_j||^2.
    lam : float, default 1.0. Ridge regulariser on alpha.

    Returns
    -------
    RichResult with payload keys estimate (mean of f-hat), alpha, beta, se,
    h, n, method.

    References
    ----------
    Gianola, D., & van Kaam, J. B. C. H. M. (2008). RKHS regression methods
        for genomic-assisted prediction. Genetics, 178(4), 2289-2303.
    Montesinos Lopez et al. (2022), Ch. 5.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m) with n matching len(y)")
    sq_norm = np.sum(M ** 2, axis=1)
    D2 = sq_norm[:, None] + sq_norm[None, :] - 2.0 * (M @ M.T)
    D2 = np.maximum(D2, 0.0)
    if h is None:
        iu = np.triu_indices(n, k=1)
        med = float(np.median(D2[iu])) if iu[0].size else 1.0
        h = med if med > 0 else 1.0
    K = np.exp(-D2 / h)
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    cand = np.column_stack([np.ones(n), Xa]) if Xa.size else np.ones((n, 1))
    _, r_ = np.linalg.qr(cand)
    keep = np.where(np.abs(np.diag(r_)) > 1e-10)[0]
    X = cand[:, keep] if keep.size else np.ones((n, 1))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    A = K + lam * np.eye(n)
    alpha = np.linalg.solve(A, r)
    f_hat = K @ alpha
    y_hat = X @ beta + f_hat
    resid = y - y_hat
    se = float(np.sqrt(np.sum(resid ** 2) / max(n - X.shape[1], 1)))
    estimate = float(np.mean(f_hat))
    return RichResult(
        title="RKHS (Gaussian kernel) regression",
        summary_lines=[
            ("n", n),
            ("bandwidth h", h),
            ("lambda", lam),
            ("mean f_hat", estimate),
            ("residual SE", se),
        ],
        payload={
            "estimate": estimate,
            "alpha": alpha,
            "beta": beta,
            "K": K,
            "f_hat": f_hat,
            "se": se,
            "h": h,
            "n": n,
            "method": "RKHS regression (Gaussian kernel)",
        },
    )


def cheatsheet():
    return "rkhsf: RKHS regression with Gaussian kernel for genomics"


# CANONICAL TEST
# np.random.seed(1); M = np.random.randint(0,3,(5,4)).astype(float)
# y = np.array([1.0,2.0,1.5,2.5,2.0]); x = np.zeros(5)
# r = rkhs_full(x, y, M); 0 < r.h < inf; alpha is length 5.
