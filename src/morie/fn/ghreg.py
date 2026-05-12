# morie.fn — function file (hadesllm/morie)
"""Bayesian-nonparametric regression — GP prior on the mean."""
import numpy as np
from ._richresult import RichResult
from .ghgps import ghosal_gp_squared_exponential

__all__ = ["ghosal_np_regression"]


def ghosal_np_regression(x, y, length_scale=None, sigma_f=1.0, noise=None):
    """Bayesian-nonparametric regression: ``Y = f(X) + eps``, ``f ~ GP``.

    Wraps the SE-kernel GP from :func:`ghosal_gp_squared_exponential`.
    Reports the posterior-mean prediction at the training inputs plus
    a posterior 95% credible band, an in-sample R^2, and the GP
    marginal log-likelihood.

    Posterior-predictive::
        y_star | y ~ N(mu_star, sd_star^2 + noise^2).

    Parameters
    ----------
    x : (n,) or (n, d).
    y : (n,).
    length_scale, sigma_f, noise : kernel hyper-parameters (defaults
        as in :func:`ghosal_gp_squared_exponential`).

    Returns
    -------
    RichResult with ``estimate`` (mean prediction), ``se`` (mean
    posterior sd), ``r2``, ``log_marginal``, ``ci_lower``,
    ``ci_upper``.

    References
    ----------
    Rasmussen & Williams (2006). GPML.
    Ghosal & van der Vaart (2017) Ch 12.
    """
    gp = ghosal_gp_squared_exponential(
        x, y, length_scale=length_scale, sigma_f=sigma_f, noise=noise)
    y_arr = np.asarray(y, dtype=float).ravel()
    mu = np.asarray(gp["mu"])
    sd = np.asarray(gp["sd"])
    n = int(y_arr.size)
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    ss_res = float(np.sum((y_arr - mu) ** 2))
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    # Marginal log-likelihood under a Gaussian likelihood
    sigma_n = float(gp["noise"])
    var_pred = sd ** 2 + sigma_n ** 2
    log_marg = float(-0.5 * np.sum((y_arr - mu) ** 2 / var_pred
                                    + np.log(2 * np.pi * var_pred)))
    ci_lower = (mu - 1.96 * np.sqrt(sd ** 2 + sigma_n ** 2)).tolist()
    ci_upper = (mu + 1.96 * np.sqrt(sd ** 2 + sigma_n ** 2)).tolist()
    return RichResult(payload={
        "estimate": float(np.mean(mu)),
        "se": float(np.mean(sd)),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "r2": float(r2),
        "log_marginal": log_marg,
        "length_scale": float(gp["length_scale"]),
        "noise": sigma_n,
        "n": n,
        "method": "GP regression posterior",
    })


def cheatsheet():
    return "ghreg: Bayesian nonparametric regression"


# CANONICAL TEST
# >>> import numpy as np
# >>> from morie.fn.ghreg import ghosal_np_regression
# >>> rng = np.random.default_rng(0)
# >>> x = np.linspace(0, 1, 30); y = np.sin(np.pi*x) + 0.05*rng.normal(size=30)
# >>> r = ghosal_np_regression(x, y)
# >>> r["r2"] > 0.8
# True
