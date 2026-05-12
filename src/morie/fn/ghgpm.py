# morie.fn -- function file (hadesllm/morie)
"""Gaussian-process regression -- Matern kernel."""
import numpy as np
from scipy.special import gamma as gammafn, kv
from ._richresult import RichResult

__all__ = ["ghosal_gp_matern"]


def ghosal_gp_matern(x, y, nu=1.5, length_scale=None, sigma_f=1.0,
                      noise=None, x_star=None):
    """Posterior mean of a GP with Matern kernel.

    Kernel::
        k(r) = sigma_f^2 * 2^{1-nu}/Gamma(nu) *
               (sqrt(2 nu) r / l)^nu * K_nu(sqrt(2 nu) r / l).

    Closed forms are used for half-integer ``nu`` in ``{0.5, 1.5, 2.5}``;
    for other ``nu`` the Bessel form is computed via
    ``scipy.special.kv``.

    Returns
    -------
    RichResult with ``estimate``, ``se``, ``mu``, ``sd``,
    ``length_scale``, ``nu``.

    References
    ----------
    Stein, M. (1999). Interpolation of Spatial Data.
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
    sq = (np.sum(x ** 2, axis=1, keepdims=True)
          + np.sum(x ** 2, axis=1)[None, :] - 2 * x @ x.T)
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
        r = np.sqrt(sq_ab)
        if nu == 0.5:
            return sigma_f ** 2 * np.exp(-r / length_scale)
        if nu == 1.5:
            t = np.sqrt(3.0) * r / length_scale
            return sigma_f ** 2 * (1 + t) * np.exp(-t)
        if nu == 2.5:
            t = np.sqrt(5.0) * r / length_scale
            return sigma_f ** 2 * (1 + t + t ** 2 / 3) * np.exp(-t)
        rr = np.where(r > 1e-12, r, 1e-12)
        z = np.sqrt(2.0 * nu) * rr / length_scale
        coef = sigma_f ** 2 * (2.0 ** (1.0 - nu)) / gammafn(nu)
        K = coef * (z ** nu) * kv(nu, z)
        K = np.where(r > 1e-12, K, sigma_f ** 2)
        return K

    K = kernel(x, x) + noise ** 2 * np.eye(n)
    K_s = kernel(x_star, x)
    K_ss_diag = sigma_f ** 2 * np.ones(x_star.shape[0])
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    alpha_ = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu = K_s @ alpha_
    v = np.linalg.solve(L, K_s.T)
    var = K_ss_diag - np.sum(v ** 2, axis=0)
    sd = np.sqrt(np.clip(var, 0, None))
    return RichResult(payload={
        "estimate": float(np.mean(mu)),
        "se": float(np.mean(sd)),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "length_scale": float(length_scale),
        "nu": float(nu),
        "noise": float(noise),
        "n": n,
        "method": "GP regression (Matern kernel)",
    })


def cheatsheet():
    return "ghgpm: GP with Matern kernel"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghgpm import ghosal_gp_matern
# >>> rng = np.random.default_rng(0)
# >>> x = np.linspace(0, 1, 20); y = x**2 + 0.05*rng.normal(size=20)
# >>> r = ghosal_gp_matern(x, y, nu=1.5)
# >>> r["se"] >= 0
# True
