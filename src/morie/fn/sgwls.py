"""Weighted least squares variogram fitting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def wls_variogram_fit(emp_gamma, lags, model="spherical", counts=None):
    """Fit a theoretical variogram model to empirical data via WLS.

    Weights are N(h)/gamma(h)^2 (Cressie, 1985).

    .. epigraph:: You have power over your mind, not outside events. -- Marcus Aurelius

    Parameters
    ----------
    emp_gamma : array_like
        Empirical semivariance values.
    lags : array_like
        Lag distances.
    model : str
        ``'spherical'``, ``'exponential'``, or ``'gaussian'``.
    counts : array_like, optional
        Number of pairs per lag bin (for weighting).

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.optimize import minimize

    emp = np.asarray(emp_gamma, dtype=np.float64)
    h = np.asarray(lags, dtype=np.float64)
    valid = emp > 0
    emp = emp[valid]
    h = h[valid]

    if counts is not None:
        w = np.asarray(counts, dtype=np.float64)[valid[: len(counts)] if len(counts) >= len(valid) else valid]
    else:
        w = np.ones(len(h))
    w = w[: len(h)]
    w = w / (emp**2 + 1e-10)

    def _model_fn(params, h_):
        c0, c1, a = params
        s = c0 + c1
        if model == "spherical":
            return np.where(h_ <= a, c0 + c1 * (1.5 * h_ / a - 0.5 * (h_ / a) ** 3), s)
        elif model == "exponential":
            return c0 + c1 * (1.0 - np.exp(-h_ / a))
        else:
            return c0 + c1 * (1.0 - np.exp(-((h_ / a) ** 2)))

    def _objective(params):
        pred = _model_fn(params, h)
        return np.sum(w * (emp - pred) ** 2)

    c0_init = max(0, emp[0] * 0.5)
    c1_init = emp.max() - c0_init
    a_init = h[len(h) // 2]

    result = minimize(
        _objective,
        [c0_init, c1_init, a_init],
        bounds=[(0, None), (0, None), (1e-6, None)],
        method="L-BFGS-B",
    )

    nugget, partial_sill, range_p = result.x
    fitted = _model_fn(result.x, h)
    rmse = float(np.sqrt(np.mean((emp - fitted) ** 2)))

    return DescriptiveResult(
        name="wls_variogram_fit",
        value=rmse,
        extra={
            "nugget": float(nugget),
            "partial_sill": float(partial_sill),
            "sill": float(nugget + partial_sill),
            "range": float(range_p),
            "model": model,
            "rmse": rmse,
            "fitted_values": fitted.tolist(),
        },
    )


sgwls = wls_variogram_fit


def cheatsheet() -> str:
    return "wls_variogram_fit({}) -> Weighted least squares variogram fitting."
