# morie.fn — function file (hadesllm/morie)
"""Bayesian-nonparametric classification — probit-link GP."""
import numpy as np
from scipy.stats import norm
from ._richresult import RichResult

__all__ = ["ghosal_np_classification"]


def ghosal_np_classification(x, y, length_scale=None, sigma_f=1.0,
                              n_iter=300, seed=0):
    """Probit-link GP classifier with Laplace-approximation posterior.

    Model::
        P(Y_i = 1 | X_i) = Phi(f(X_i)),  f ~ GP(0, k).

    The Laplace-approximation (Rasmussen & Williams Ch 3.4) finds the
    mode of the log-posterior of f and approximates the posterior by a
    Gaussian centred there with the negative Hessian as precision.

    Parameters
    ----------
    x : (n,) or (n,d).
    y : (n,) array of {0,1}.
    length_scale, sigma_f : kernel.
    n_iter : Newton iterations.

    Returns
    -------
    RichResult with ``estimate`` (mean of posterior class-1
    probabilities at training points), ``p_hat`` array, ``accuracy``
    (in-sample), ``log_marginal``.

    References
    ----------
    Rasmussen & Williams (2006). GPML, Ch 3.
    Ghosal & van der Vaart (2017) Ch 12.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float).ravel()
    y_pm = 2 * y - 1  # to {-1, +1}
    n = x.shape[0]
    sq = (np.sum(x ** 2, axis=1, keepdims=True)
          + np.sum(x ** 2, axis=1)[None, :] - 2 * x @ x.T)
    sq = np.clip(sq, 0, None)
    if length_scale is None:
        d = np.sqrt(sq[np.triu_indices_from(sq, k=1)])
        length_scale = float(np.median(d[d > 0])) if d.size and (d > 0).any() else 1.0
        length_scale = max(length_scale, 1e-3)
    K = sigma_f ** 2 * np.exp(-sq / (2 * length_scale ** 2)) + 1e-6 * np.eye(n)

    # Newton-Raphson on f for probit likelihood
    f = np.zeros(n)
    for _ in range(n_iter):
        z = y_pm * f
        phi = norm.pdf(z)
        Phi = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
        # gradient of log-likelihood
        grad_ll = y_pm * phi / Phi
        # negative Hessian (W = diag, positive)
        W = (phi / Phi) * (phi / Phi + z)
        W = np.clip(W, 1e-8, None)
        sW = np.sqrt(W)
        B = np.eye(n) + (sW[:, None] * K) * sW[None, :]
        try:
            L = np.linalg.cholesky(B)
        except np.linalg.LinAlgError:
            break
        b = W * f + grad_ll
        a = b - sW * np.linalg.solve(L.T, np.linalg.solve(L, sW * (K @ b)))
        f_new = K @ a
        if np.max(np.abs(f_new - f)) < 1e-6:
            f = f_new
            break
        f = f_new
    p_hat = norm.cdf(f)
    # Approximate log-marginal (Laplace)
    z = y_pm * f
    Phi = np.clip(norm.cdf(z), 1e-12, 1 - 1e-12)
    ll = float(np.sum(np.log(Phi)))
    try:
        log_marg = ll - 0.5 * float(f @ np.linalg.solve(K, f)) \
            - 0.5 * float(np.linalg.slogdet(np.eye(n) + (sW[:, None] * K) * sW[None, :])[1])
    except np.linalg.LinAlgError:
        log_marg = float("nan")
    pred = (p_hat >= 0.5).astype(int)
    accuracy = float(np.mean(pred == y))
    return RichResult(payload={
        "estimate": float(np.mean(p_hat)),
        "p_hat": p_hat.tolist(),
        "accuracy": accuracy,
        "log_marginal": log_marg,
        "length_scale": float(length_scale),
        "n": n,
        "method": "Probit-link GP classifier (Laplace)",
    })


def cheatsheet():
    return "ghcls: Bayesian nonparametric classification"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghcls import ghosal_np_classification
# >>> rng = np.random.default_rng(0)
# >>> x = rng.normal(size=(40, 2)); y = (x[:,0] > 0).astype(int)
# >>> r = ghosal_np_classification(x, y)
# >>> r["accuracy"] > 0.7
# True
