# morie.fn -- function file (hadesllm/morie)
"""Gaussian-process regression -- squared-exponential kernel."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_gp_squared_exponential"]


def ghosal_gp_squared_exponential(x, y, length_scale=None, sigma_f=1.0,
                                    noise=None, x_star=None):
    """Posterior mean of a Gaussian process with SE kernel.

    Kernel::
        k(x, x') = sigma_f^2 exp(-||x - x'||^2 / (2 l^2)).

    Posterior::
        f_star | y ~ N(K_star (K + sigma_n^2 I)^{-1} y,
                       K_starstar - K_star (K + sigma_n^2 I)^{-1} K_star^T).

    Parameters
    ----------
    x : array-like (n,) or (n, d)
        Training inputs.
    y : array-like (n,)
        Training targets.
    length_scale : float or None
        Defaults to the median pairwise distance (Gretton et al. 2012
        heuristic).
    sigma_f : float
        Signal sd.
    noise : float or None
        Observation sd.  Defaults to ``0.1 * sd(y)``.
    x_star : array-like or None
        Test inputs.  If ``None``, posterior is evaluated at the
        training inputs (in-sample fit + posterior sd).

    Returns
    -------
    RichResult with ``estimate`` (mean posterior prediction at training
    inputs), ``se`` (mean posterior sd), ``mu`` and ``sd`` arrays.

    References
    ----------
    Rasmussen & Williams (2006). Gaussian Processes for ML.
    Ghosal & van der Vaart (2017) Ch 7.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if x_star is None:
        x_star = x
    x_star = np.asarray(x_star, dtype=float)
    if x_star.ndim == 1:
        x_star = x_star[:, None]
    sq = np.sum(x ** 2, axis=1, keepdims=True) + \
        np.sum(x ** 2, axis=1)[None, :] - 2 * x @ x.T
    sq = np.clip(sq, 0, None)
    if length_scale is None:
        d = np.sqrt(sq[np.triu_indices_from(sq, k=1)])
        length_scale = float(np.median(d[d > 0])) if d.size and (d > 0).any() else 1.0
        length_scale = max(length_scale, 1e-3)
    if noise is None:
        noise = 0.1 * float(np.std(y, ddof=1)) if n > 1 else 0.1
        noise = max(noise, 1e-3)

    def kernel(a, b):
        sq_ab = (np.sum(a ** 2, axis=1, keepdims=True)
                 + np.sum(b ** 2, axis=1)[None, :] - 2 * a @ b.T)
        sq_ab = np.clip(sq_ab, 0, None)
        return sigma_f ** 2 * np.exp(-sq_ab / (2 * length_scale ** 2))

    K = kernel(x, x) + noise ** 2 * np.eye(n)
    K_s = kernel(x_star, x)
    K_ss_diag = sigma_f ** 2 * np.ones(x_star.shape[0])
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = K_s @ alpha
    v = np.linalg.solve(L, K_s.T)
    var = K_ss_diag - np.sum(v ** 2, axis=0)
    sd = np.sqrt(np.clip(var, 0, None))
    return RichResult(payload={
        "estimate": float(np.mean(mu)),
        "se": float(np.mean(sd)),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "length_scale": float(length_scale),
        "noise": float(noise),
        "n": n,
        "method": "GP regression (squared-exponential kernel)",
    })


def cheatsheet():
    return "ghgps: GP with squared exponential kernel"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghgps import ghosal_gp_squared_exponential
# >>> rng = np.random.default_rng(0)
# >>> x = np.linspace(0, 1, 20); y = np.sin(2*np.pi*x) + 0.05*rng.normal(size=20)
# >>> r = ghosal_gp_squared_exponential(x, y)
# >>> r["se"] >= 0
# True
