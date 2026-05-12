# morie.fn — function file (hadesllm/morie)
"""Marker variance-component estimation (per-marker sigma_m^2)."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .gmatv import grm_vanraden

__all__ = ["marker_variance"]


def marker_variance(x, y, markers):
    """Estimate the per-marker variance component sigma_m^2.

    Two routes are reported:

    1) Variance-component (REML-style) split via VanRaden's identity::

           sigma_g^2 = var(g_hat),  sigma_m^2 = sigma_g^2 / (2 sum p_j q_j)

       where p_j q_j is the j-th allele-frequency product.  We use a
       quick BLUP fit (lambda = sigma_e^2/sigma_g^2 = 1) to obtain g_hat.

    2) The naive Montesinos formula::

           sigma_m^2 ≈ sigma_g^2 / p

       reported alongside for direct comparison.

    Parameters
    ----------
    x : array-like (n,) or (n,q). Fixed effects (intercept always added).
    y : array-like (n,)
    markers : array-like (n,m). Coded 0/1/2.

    Returns
    -------
    RichResult with payload keys estimate (sigma_m^2 VanRaden),
    sigma_g2, sigma_e2, sigma_m2_vanraden, sigma_m2_naive,
    sum_2pq, p_freq, n, p, method.

    References
    ----------
    VanRaden, P. M. (2008). Efficient methods to compute genomic predictions.
        J Dairy Sci, 91(11), 4414-4423.
    Montesinos Lopez et al. (2022), Ch. 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m)")
    n_, m_ = M.shape
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    cand = np.column_stack([np.ones(n), Xa]) if Xa.size else np.ones((n, 1))
    _, r_ = np.linalg.qr(cand)
    keep = np.where(np.abs(np.diag(r_)) > 1e-10)[0]
    X = cand[:, keep] if keep.size else np.ones((n, 1))
    p_freq = M.mean(axis=0) / 2.0
    sum_2pq = float(2.0 * np.sum(p_freq * (1.0 - p_freq)))
    sum_2pq = sum_2pq if sum_2pq > 0 else 1.0
    G = grm_vanraden(M, method=1).estimate + 1e-6 * np.eye(n)
    G_inv = np.linalg.inv(G)
    # GBLUP with lambda = 1 (heritability proxy)
    p_x = X.shape[1]
    lam = 1.0
    C = np.block([[X.T @ X,           X.T               ],
                  [X,                 np.eye(n) + lam * G_inv]])
    rhs = np.concatenate([X.T @ y, y])
    sol = np.linalg.solve(C, rhs)
    beta = sol[:p_x]; g_hat = sol[p_x:]
    sigma_g2 = float(np.var(g_hat, ddof=1)) if n > 1 else 0.0
    resid = y - X @ beta - g_hat
    sigma_e2 = float(np.var(resid, ddof=1)) if n > 1 else 0.0
    sigma_m2_vanraden = sigma_g2 / sum_2pq
    sigma_m2_naive = sigma_g2 / max(m_, 1)
    h2 = sigma_g2 / (sigma_g2 + sigma_e2) if (sigma_g2 + sigma_e2) > 0 else float("nan")
    return RichResult(
        title="Marker variance component",
        summary_lines=[
            ("n", n),
            ("m (markers)", m_),
            ("sigma_g^2", sigma_g2),
            ("sigma_e^2", sigma_e2),
            ("h^2 (heritability)", h2),
            ("sigma_m^2 (VanRaden)", sigma_m2_vanraden),
            ("sigma_m^2 (naive σ_g²/p)", sigma_m2_naive),
            ("sum 2 p_j q_j", sum_2pq),
        ],
        payload={
            "estimate": sigma_m2_vanraden,
            "sigma_g2": sigma_g2,
            "sigma_e2": sigma_e2,
            "h2": h2,
            "sigma_m2_vanraden": sigma_m2_vanraden,
            "sigma_m2_naive": sigma_m2_naive,
            "sum_2pq": sum_2pq,
            "p_freq": p_freq,
            "n": n,
            "p": m_,
            "method": "VanRaden + naive marker-variance split",
        },
    )


def cheatsheet():
    return "mrkvr: Marker variance-component estimation"


# CANONICAL TEST
# np.random.seed(16); M = np.random.randint(0,3,(20,8)).astype(float)
# y = M @ np.random.randn(8) + 0.5*np.random.randn(20); x = np.zeros(20)
# r = marker_variance(x, y, M); 0 < r.h2 < 1; sigma_m^2 finite.
