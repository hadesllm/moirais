# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Bayesian prior elicitation. 'The wisdom of ages.' -- Alpha Trion"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from ._containers import DescriptiveResult


def prior_elicit(
    quantiles: dict[float, float],
    *,
    family: str = "normal",
) -> DescriptiveResult:
    """Elicit Bayesian prior parameters from expert quantile judgments.

    Given a set of {probability: value} quantile pairs, finds the
    distribution parameters that best match the stated quantiles.

    Parameters
    ----------
    quantiles : dict
        {probability: value} pairs, e.g. {0.05: 10, 0.50: 50, 0.95: 90}.
    family : str
        Distribution family: 'normal', 'lognormal', 'beta', 'gamma'.

    Returns
    -------
    DescriptiveResult
        With ``value`` = fitted parameters and ``extra`` containing fit error.
    """
    if len(quantiles) < 2:
        raise ValueError("Need at least 2 quantile-value pairs")

    probs = np.array(sorted(quantiles.keys()))
    vals = np.array([quantiles[p] for p in probs])

    if family == "normal":

        def loss(params):
            mu, sigma = params
            if sigma <= 0:
                return 1e10
            predicted = stats.norm.ppf(probs, mu, sigma)
            return np.sum((predicted - vals) ** 2)

        x0 = [vals.mean(), max(vals.std(), 0.1)]
        res = optimize.minimize(loss, x0, method="Nelder-Mead")
        params = {"mu": float(res.x[0]), "sigma": float(abs(res.x[1]))}

    elif family == "lognormal":
        if np.any(vals <= 0):
            raise ValueError("lognormal requires positive quantile values")
        log_vals = np.log(vals)

        def loss(params):
            mu, sigma = params
            if sigma <= 0:
                return 1e10
            predicted = np.exp(stats.norm.ppf(probs, mu, sigma))
            return np.sum((predicted - vals) ** 2)

        x0 = [log_vals.mean(), max(log_vals.std(), 0.1)]
        res = optimize.minimize(loss, x0, method="Nelder-Mead")
        params = {"mu": float(res.x[0]), "sigma": float(abs(res.x[1]))}

    elif family == "beta":
        if np.any(vals < 0) or np.any(vals > 1):
            raise ValueError("beta requires values in [0, 1]")

        def loss(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            predicted = stats.beta.ppf(probs, a, b)
            return np.sum((predicted - vals) ** 2)

        x0 = [2.0, 2.0]
        res = optimize.minimize(loss, x0, method="Nelder-Mead")
        params = {"alpha": float(max(res.x[0], 0.01)), "beta": float(max(res.x[1], 0.01))}

    elif family == "gamma":
        if np.any(vals <= 0):
            raise ValueError("gamma requires positive values")

        def loss(params):
            shape, scale = params
            if shape <= 0 or scale <= 0:
                return 1e10
            predicted = stats.gamma.ppf(probs, shape, scale=scale)
            return np.sum((predicted - vals) ** 2)

        x0 = [2.0, vals.mean() / 2]
        res = optimize.minimize(loss, x0, method="Nelder-Mead")
        params = {"shape": float(max(res.x[0], 0.01)), "scale": float(max(res.x[1], 0.01))}

    else:
        raise ValueError(f"Unknown family: {family}")

    return DescriptiveResult(
        name="prior_elicit",
        value=params,
        extra={
            "family": family,
            "fit_error": float(res.fun),
            "n_quantiles": len(quantiles),
            "input_quantiles": quantiles,
        },
    )


alpht = prior_elicit


def cheatsheet() -> str:
    return "prior_elicit({}) -> Bayesian prior elicitation. 'The wisdom of ages.' -- Alpha T"
