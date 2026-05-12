# morie.fn — function file (hadesllm/morie)
"""Log-density estimation by exponential-family expansion."""
import numpy as np
from scipy.optimize import minimize
from ._richresult import RichResult

__all__ = ["ghosal_log_density"]


def ghosal_log_density(x, K=5, grid=None):
    """Log-spline / Lenth-type log-density estimator.

    Parameterise

        log f(x) = sum_{k=1}^K theta_k psi_k(x) - log Z(theta),

    with monomial basis ``psi_k(x) = x^k / sigma_x^k`` (the simplest
    Stone–Koo log-spline, Ghosal Ch 8).  The MLE solves

        theta_hat = argmax  sum_i sum_k theta_k psi_k(X_i)
                            - n log integral exp(sum theta_k psi_k(u)) du.

    We optimise this directly by L-BFGS-B; the integral is approximated
    by trapezoidal quadrature on a 401-point grid spanning
    ``mean ± 5 sd``.  Returns the fitted log-density at the sample mean.

    Parameters
    ----------
    x : array-like.
    K : int — basis size.
    grid : array-like or None.

    Returns
    -------
    RichResult with ``estimate`` (fitted log-density at sample mean),
    ``theta`` (basis coefficients), ``log_lik``,
    ``grid``, ``log_density`` array, ``K``.

    References
    ----------
    Stone, C. (1990). Large-sample inference for log-spline models. AOS 18.
    Lenth (1983). Some properties of U-shaped detection. JASA.
    Ghosal & van der Vaart (2017) Ch 8.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = int(x.size)
    if n < 5:
        return RichResult(payload={
            "estimate": float("nan"), "n": n,
            "method": "Log-density (n<5)",
        })
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    s = max(s, 1e-6)
    z = (x - m) / s
    if grid is None:
        gz = np.linspace(z.min() - 1.0, z.max() + 1.0, 401)
    else:
        gz = (np.asarray(grid, dtype=float) - m) / s
    # Use Hermite-style monomial basis (centred, scaled).
    def basis(u):
        return np.stack([u ** (k + 1) for k in range(K)], axis=-1)

    Bx = basis(z)              # (n, K)
    Bg = basis(gz)             # (G, K)

    def neg_ll(theta):
        eta_x = Bx @ theta
        eta_g = Bg @ theta
        # Stable log-sum: subtract max
        M = float(np.max(eta_g))
        Z = M + np.log(np.trapezoid(np.exp(eta_g - M), gz))
        return float(-np.sum(eta_x) + n * Z)

    theta0 = np.zeros(K)
    # Penalty to keep coefficients tame (small L2)
    opt = minimize(lambda t: neg_ll(t) + 1e-4 * np.sum(t ** 2),
                    theta0, method="L-BFGS-B")
    theta = opt.x
    eta_g = Bg @ theta
    M = float(np.max(eta_g))
    logZ = M + float(np.log(np.trapezoid(np.exp(eta_g - M), gz)))
    log_density = eta_g - logZ - np.log(s)   # change-of-variables back
    # log-density at sample mean
    eta0 = float(basis(np.array([0.0]))[0] @ theta)
    estimate = float(eta0 - logZ - np.log(s))
    log_lik = float(-opt.fun + 1e-4 * np.sum(theta ** 2))   # remove penalty
    return RichResult(payload={
        "estimate": estimate,
        "theta": theta.tolist(),
        "log_lik": log_lik,
        "grid": (gz * s + m).tolist(),
        "log_density": log_density.tolist(),
        "K": int(K),
        "n": n,
        "method": "Log-spline density (Stone 1990)",
    })


def cheatsheet():
    return "ghlgd: Log-density estimation"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghlgd import ghosal_log_density
# >>> r = ghosal_log_density(np.random.default_rng(0).normal(size=200), K=3)
# >>> r["estimate"] < 0
# True
