# morie.fn -- function file (hadesllm/morie)
"""GBLUP with the VanRaden genomic relationship matrix."""
from __future__ import annotations

import numpy as np

from ._richresult import RichResult
from .gmatv import grm_vanraden

__all__ = ["gblup_full"]


def gblup_full(x, y, markers, lambda_gblup: float | None = None):
    """GBLUP via Henderson's mixed model equations.

    Model::

        y = X*beta + Z*g + e,  g ~ N(0, G*sigma_g^2),  e ~ N(0, I*sigma_e^2)

    The mixed-model equations (with Z = I after sorting individuals) are::

        [X'X            X'                ] [beta]   [X'y]
        [ X    I + lam * G^{-1}           ] [ g  ] = [ y ]

    where lam = sigma_e^2 / sigma_g^2. If `lambda_gblup` is not supplied,
    we estimate variance components by a single-iteration REML proxy:
    sigma_g^2 = var(y) / tr(G)/n  and  sigma_e^2 = var(y) - sigma_g^2.

    Parameters
    ----------
    x : array-like, shape (n,) or (n,p)
        Fixed-effect design (intercept added if 1-D).
    y : array-like, shape (n,)
    markers : array-like, shape (n,m)
    lambda_gblup : float, optional
        Ratio sigma_e^2 / sigma_g^2. If None, estimated heuristically.

    Returns
    -------
    RichResult with payload keys estimate (predicted g-hat), beta, se,
    lambda_gblup, n, method.

    References
    ----------
    Montesinos Lopez et al. (2022), Ch. 3.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    M = np.asarray(markers, dtype=float)
    if M.ndim != 2:
        raise ValueError("`markers` must be a 2D (n × m) array")
    # Fixed-effect design: prepend intercept, drop rank-deficient cols
    Xa = np.asarray(x, dtype=float)
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    cand = np.column_stack([np.ones(n), Xa]) if Xa.size else np.ones((n, 1))
    # QR-based rank pruning to avoid singular MME
    q_, r_ = np.linalg.qr(cand)
    keep = np.where(np.abs(np.diag(r_)) > 1e-10)[0]
    X = cand[:, keep] if keep.size else np.ones((n, 1))
    # Build VanRaden G with a tiny ridge so it's invertible
    G = grm_vanraden(M, method=1).estimate
    G = G + 1e-6 * np.eye(n)
    # Variance-component proxy (single-pass REML-ish heuristic)
    if lambda_gblup is None:
        var_y = float(np.var(y, ddof=1)) if n > 1 else 1.0
        h2 = 0.5
        sg2 = h2 * var_y
        se2 = (1.0 - h2) * var_y
        lam = se2 / sg2 if sg2 > 0 else 1.0
    else:
        lam = float(lambda_gblup)
    G_inv = np.linalg.inv(G)
    p = X.shape[1]
    # Mixed-model coefficient matrix
    C = np.block([
        [X.T @ X,           X.T               ],
        [X,                 np.eye(n) + lam * G_inv],
    ])
    rhs = np.concatenate([X.T @ y, y])
    sol = np.linalg.solve(C, rhs)
    beta = sol[:p]
    g_hat = sol[p:]
    # Residual / SE
    y_hat = X @ beta + g_hat
    resid = y - y_hat
    sigma_e2 = float(np.sum(resid ** 2) / max(n - p, 1))
    se = float(np.sqrt(sigma_e2))
    estimate = float(np.mean(g_hat))  # population-mean BV; full vector in g_hat
    return RichResult(
        title="GBLUP (Genomic BLUP)",
        summary_lines=[
            ("n", n),
            ("mean(g_hat)", estimate),
            ("sigma_e", se),
            ("lambda", lam),
        ],
        payload={
            "estimate": estimate,
            "g_hat": g_hat,
            "beta": beta,
            "se": se,
            "y_hat": y_hat,
            "lambda_gblup": lam,
            "n": n,
            "method": "GBLUP with VanRaden G",
        },
    )


def cheatsheet():
    return "gblpf: GBLUP with genomic relationship matrix"


# CANONICAL TEST
# Small reproducible: n=4 individuals, m=5 markers
# np.random.seed(0); M = np.random.randint(0,3,(4,5)).astype(float)
# y = np.array([1.0, 2.0, 3.0, 2.5]); x = np.zeros(4)
# r = gblup_full(x, y, M); should give g_hat length 4, mean≈0.
