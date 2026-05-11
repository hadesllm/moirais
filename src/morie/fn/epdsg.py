# morie.fn — function file (hadesllm/morie)
"""Epidemic curve fitting (log-normal / gamma peak models)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as _st
from scipy.optimize import minimize


def epidemic_curve_fit(
    incidence: np.ndarray,
    *,
    distribution: str = "lognormal",
    t: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a parametric distribution to an epidemic curve.

    Fits a scaled log-normal or gamma density to daily incidence to
    estimate peak timing, shape, and total size.

    Parameters
    ----------
    incidence : array_like
        Daily incidence counts.
    distribution : str, default "lognormal"
        "lognormal" or "gamma".
    t : array_like or None
        Time points. If None, uses 0, 1, 2, ...

    Returns
    -------
    dict
        Keys: 'peak_time', 'peak_height', 'total_cases', 'params',
              'fitted_curve', 'distribution', 'rmse'.

    References
    ----------
    Becker, N. G. & Britton, T. (1999). Statistical studies of
    infectious disease incidence. Journal of the Royal Statistical
    Society B, 61(2), 287-307.
    """
    inc = np.asarray(incidence, dtype=float)
    if inc.ndim != 1 or inc.size < 5:
        raise ValueError("incidence must be 1-D with >= 5 points.")

    if t is None:
        t = np.arange(len(inc), dtype=float)
    else:
        t = np.asarray(t, dtype=float)

    total = float(np.sum(inc))
    t_pos = t[t > 0]
    inc_pos = inc[t > 0]

    if distribution == "lognormal":
        def _model(params):
            amp, mu, sigma = params
            if sigma <= 0 or amp <= 0:
                return 1e12
            curve = amp * _st.lognorm.pdf(t, s=sigma, scale=np.exp(mu))
            return float(np.sum((inc - curve) ** 2))

        mu0 = np.log(float(t[np.argmax(inc)]) + 1)
        x0 = [total, mu0, 0.5]
        res = minimize(_model, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8})
        amp, mu, sigma = res.x
        fitted = amp * _st.lognorm.pdf(t, s=abs(sigma), scale=np.exp(mu))
        params = {"amplitude": float(amp), "mu": float(mu),
                  "sigma": float(abs(sigma))}

    elif distribution == "gamma":
        def _model(params):
            amp, a, scale = params
            if a <= 0 or scale <= 0 or amp <= 0:
                return 1e12
            curve = amp * _st.gamma.pdf(t, a=a, scale=scale)
            return float(np.sum((inc - curve) ** 2))

        pk = float(t[np.argmax(inc)]) + 1
        x0 = [total, 2.0, pk / 2]
        res = minimize(_model, x0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8})
        amp, a, scale = res.x
        fitted = abs(amp) * _st.gamma.pdf(t, a=abs(a), scale=abs(scale))
        params = {"amplitude": float(abs(amp)), "shape": float(abs(a)),
                  "scale": float(abs(scale))}
    else:
        raise ValueError("distribution must be 'lognormal' or 'gamma'.")

    peak_idx = int(np.argmax(fitted))
    rmse = float(np.sqrt(np.mean((inc - fitted) ** 2)))

    return {
        "peak_time": float(t[peak_idx]),
        "peak_height": float(fitted[peak_idx]),
        "total_cases": total,
        "params": params,
        "fitted_curve": fitted,
        "distribution": distribution,
        "rmse": rmse,
    }


epdsg = epidemic_curve_fit


def cheatsheet() -> str:
    return "epidemic_curve_fit({}) -> Fit log-normal/gamma to epi curve."
