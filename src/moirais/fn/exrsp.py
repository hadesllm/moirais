# moirais.fn — function file (hadesllm/moirais)
"""Exposure-response modeling for environmental epidemiology."""

import numpy as np

from ._containers import DescriptiveResult


def exposure_response(exposure, outcome, covariates=None, n_spline_knots=4):
    """
    Fit exposure-response curve via restricted cubic splines.

    :param exposure: (n,) continuous exposure variable.
    :param outcome: (n,) health outcome.
    :param covariates: (n, p) optional confounders.
    :param n_spline_knots: Number of spline knots (3-7).
    :return: DescriptiveResult with predicted curve, knot locations.

    References
    ----------
    Harrell FE (2015). Regression Modeling Strategies. 2nd ed. Springer.
    """
    exp = np.asarray(exposure, dtype=np.float64).ravel()
    y = np.asarray(outcome, dtype=np.float64).ravel()
    n = len(exp)
    knots = np.percentile(exp, np.linspace(5, 95, n_spline_knots))

    def _rcs_basis(x, knots):
        k = len(knots)
        basis = [np.ones(len(x)), x]
        for j in range(k - 2):
            t_j = knots[j]
            t_km1 = knots[-2]
            t_k = knots[-1]
            d_j = np.maximum(x - t_j, 0) ** 3
            d_km1 = np.maximum(x - t_km1, 0) ** 3
            d_k = np.maximum(x - t_k, 0) ** 3
            denom = t_k - t_km1
            if abs(denom) < 1e-10:
                denom = 1e-10
            col = (d_j - d_km1) / denom - (d_j - d_k) / denom
            basis.append(col)
        return np.column_stack(basis)

    X = _rcs_basis(exp, knots)
    if covariates is not None:
        X = np.column_stack([X, np.asarray(covariates)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    resid = y - y_pred
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)

    grid = np.linspace(exp.min(), exp.max(), 100)
    X_grid = _rcs_basis(grid, knots)
    if covariates is not None:
        X_grid = np.column_stack([X_grid, np.zeros((100, covariates.shape[1] if hasattr(covariates, "shape") else 1))])
    curve = X_grid @ beta

    return DescriptiveResult(
        name="exposure_response",
        value=float(r2),
        extra={
            "exposure_grid": grid.tolist(),
            "predicted_response": curve.tolist(),
            "knots": knots.tolist(),
            "r_squared": float(r2),
            "coefficients": beta.tolist(),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "exposure_response({}) -> Exposure-response modeling for environmental epidemiology."
