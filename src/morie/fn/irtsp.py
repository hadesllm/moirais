# morie.fn -- function file (hadesllm/morie)
"""IRT-based 2PL spatial ideal-point model (Armstrong Ch 4)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["irt_spatial", "irtsp"]


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def irt_spatial(x, n_iter: int = 60, tol: float = 1e-6):
    """2-parameter logistic (2PL) IRT ideal-point model.

    Joint-MLE alternating updates:
        P(yea_ij) = sigmoid( alpha_j * (x_i - beta_j) )
    where x_i is legislator ideal point, alpha_j item discrimination,
    beta_j item difficulty (Clinton-Jackman-Rivers 2004 in MLE form).
    Identification: x is standardised to mean 0, sd 1 each iteration.

    Parameters
    ----------
    x : (n,m) binary roll-call matrix, rows = legislators, cols = items.
        A 1-D input is interpreted as a single column.
    n_iter, tol : convergence controls.

    Returns
    -------
    RichResult with keys: x_hat, alpha, beta, loglik, n_iter
    """
    M = np.asarray(x, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n, m = M.shape
    if n < 2 or m < 1:
        return RichResult(payload={"x_hat": np.full(n, np.nan),
                                   "alpha": np.full(m, np.nan),
                                   "beta": np.full(m, np.nan),
                                   "loglik": np.nan, "n_iter": 0,
                                   "method": "irt_spatial"})
    # Init via first PCA component of vote matrix
    Mc = M - np.nanmean(M, axis=0, keepdims=True)
    Mc = np.nan_to_num(Mc)
    try:
        u, s, vt = np.linalg.svd(Mc, full_matrices=False)
        x_hat = u[:, 0] * s[0]
    except np.linalg.LinAlgError:
        x_hat = np.linspace(-1, 1, n)
    x_hat = (x_hat - x_hat.mean()) / (x_hat.std() + 1e-12)
    alpha = np.ones(m)
    beta = np.zeros(m)
    prev_ll = -np.inf
    it = 0
    for it in range(1, n_iter + 1):
        # Update item params (alpha_j, beta_j) via Newton on each column
        for j in range(m):
            yj = M[:, j]
            mask = ~np.isnan(yj)
            xj = x_hat[mask]; yjm = yj[mask]
            a, b = alpha[j], beta[j]
            for _ in range(5):
                z = a * (xj - b)
                p = _logistic(z)
                w = p * (1 - p) + 1e-9
                # gradient
                ga = np.sum((yjm - p) * (xj - b))
                gb = np.sum((yjm - p) * (-a))
                # Hessian (negative)
                Haa = -np.sum(w * (xj - b) ** 2)
                Hbb = -np.sum(w * a * a)
                Hab = -np.sum(w * (-a * (xj - b) + (yjm - p) * (-1)))
                H = np.array([[Haa, Hab], [Hab, Hbb]])
                g = np.array([ga, gb])
                try:
                    step = np.linalg.solve(H - 1e-6 * np.eye(2), g)
                except np.linalg.LinAlgError:
                    break
                a, b = a - step[0], b - step[1]
                if np.max(np.abs(step)) < tol:
                    break
            alpha[j], beta[j] = a, b
        # Update legislator ideal points
        for i in range(n):
            yi = M[i]
            mask = ~np.isnan(yi)
            aj = alpha[mask]; bj = beta[mask]; yim = yi[mask]
            xi = x_hat[i]
            for _ in range(5):
                z = aj * (xi - bj)
                p = _logistic(z)
                w = p * (1 - p) + 1e-9
                g = np.sum(aj * (yim - p))
                H = -np.sum(w * aj ** 2)
                if abs(H) < 1e-12:
                    break
                step = g / H
                xi = xi - step
                if abs(step) < tol:
                    break
            x_hat[i] = xi
        x_hat = (x_hat - x_hat.mean()) / (x_hat.std() + 1e-12)
        # Log-likelihood
        Z = alpha[None, :] * (x_hat[:, None] - beta[None, :])
        P = _logistic(Z)
        mask_full = ~np.isnan(M)
        ll = float(np.sum(np.where(mask_full,
                                   M * np.log(P + 1e-12)
                                   + (1 - M) * np.log(1 - P + 1e-12), 0.0)))
        if abs(ll - prev_ll) < tol * max(1.0, abs(prev_ll)):
            break
        prev_ll = ll
    return RichResult(
        title="IRT 2PL spatial model (Clinton-Jackman-Rivers)",
        summary_lines=[("log-lik", ll), ("n legislators", n),
                       ("m items", m), ("iter", it)],
        payload={"x_hat": x_hat, "alpha": alpha, "beta": beta,
                 "loglik": ll, "n_iter": int(it),
                 "method": "irt_spatial_2pl"},
    )


irtsp = irt_spatial


def cheatsheet():
    return "irtsp: 2PL IRT ideal-point model -- Clinton-Jackman-Rivers."


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> X_true = rng.normal(size=20)
# >>> Y = (X_true[:, None] + rng.normal(size=(20, 10)) > 0).astype(int)
# >>> r = irt_spatial(Y, n_iter=20)
# >>> assert abs(np.corrcoef(r["x_hat"], X_true)[0,1]) > 0.6
