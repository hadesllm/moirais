# morie.fn -- function file (hadesllm/morie)
"""Multi-trait GBLUP."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .gmatv import grm_vanraden

__all__ = ["multi_trait_gblup"]


def multi_trait_gblup(x, y, markers, Sigma_g: np.ndarray | None = None,
                      Sigma_e: np.ndarray | None = None):
    """Multi-trait GBLUP via vec-stacked mixed-model equations.

    Model::

        Y = X*B + Z*G + E,  vec(G) ~ N(0, Sigma_g ⊗ G_mat),
                            vec(E) ~ N(0, Sigma_e ⊗ I_n)

    With Z = I and X = 1 (intercept only by default), the BLUP of vec(G) is::

        g_hat = (Sigma_g ⊗ G_mat) (V)^{-1} vec(Y - 1*B_hat)

    where V = Sigma_g ⊗ G_mat + Sigma_e ⊗ I_n.

    If Sigma_g, Sigma_e not provided we estimate them by method-of-moments:
    Sigma_g = h^2 * cov(Y) and Sigma_e = (1-h^2) * cov(Y), with h^2 = 0.5.

    Parameters
    ----------
    x : array-like (n,) or (n, q). Treated as fixed-effect design.
    y : array-like (n, t).  Multi-trait response.
    markers : array-like (n, m).
    Sigma_g, Sigma_e : (t,t) covariance matrices, optional.

    Returns
    -------
    RichResult with payload keys estimate (mean of vec(G_hat)), G_hat, B_hat,
    Sigma_g, Sigma_e, n, t, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 10.
    """
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, t = Y.shape
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2 or M.shape[0] != n:
        raise ValueError("`markers` must be (n × m) with n matching len(y)")
    G_mat = grm_vanraden(M, method=1).estimate
    G_mat = G_mat + 1e-6 * np.eye(n)
    # Fixed effects: intercept only (estimate per-trait mean)
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    cand = np.column_stack([np.ones(n), Xa]) if Xa.size else np.ones((n, 1))
    _, r_ = np.linalg.qr(cand)
    keep = np.where(np.abs(np.diag(r_)) > 1e-10)[0]
    X = cand[:, keep] if keep.size else np.ones((n, 1))
    if Sigma_g is None or Sigma_e is None:
        h2 = 0.5
        S_y = np.cov(Y.T) if t > 1 else np.array([[float(np.var(Y, ddof=1))]])
        if S_y.ndim == 0:
            S_y = np.array([[float(S_y)]])
        Sigma_g = h2 * S_y
        Sigma_e = (1.0 - h2) * S_y
    Sigma_g = np.atleast_2d(Sigma_g)
    Sigma_e = np.atleast_2d(Sigma_e)
    # OLS for B given current variance components
    B = np.linalg.lstsq(X, Y, rcond=None)[0]  # (q, t)
    R = Y - X @ B  # residuals
    # vec-stacked GBLUP:  g_hat = (Sigma_g ⊗ G_mat) V^{-1} vec(R)
    V = np.kron(Sigma_g, G_mat) + np.kron(Sigma_e, np.eye(n))
    vec_R = R.T.reshape(-1)  # column-stack (trait-major)
    SG = np.kron(Sigma_g, G_mat)
    g_hat_vec = SG @ np.linalg.solve(V, vec_R)
    G_hat = g_hat_vec.reshape(t, n).T  # (n, t)
    estimate = float(np.mean(G_hat))
    return RichResult(
        title="Multi-trait GBLUP",
        summary_lines=[
            ("n", n),
            ("t (traits)", t),
            ("mean(G_hat)", estimate),
        ],
        sections=[{
            "title": "Sigma_g (genetic covariance)",
            "headers": [f"t{i+1}" for i in range(t)],
            "table": [[float(v) for v in row] for row in Sigma_g],
        }],
        payload={
            "estimate": estimate,
            "G_hat": G_hat,
            "B_hat": B,
            "Sigma_g": Sigma_g,
            "Sigma_e": Sigma_e,
            "n": n,
            "t": t,
            "method": "Multi-trait GBLUP (vec-stacked MME)",
        },
    )


def cheatsheet():
    return "mtgbl: Multi-trait GBLUP"


# CANONICAL TEST
# np.random.seed(5); M = np.random.randint(0,3,(6,8)).astype(float)
# Y = np.random.randn(6, 2); x = np.zeros(6)
# r = multi_trait_gblup(x, Y, M); r.G_hat.shape == (6,2).
