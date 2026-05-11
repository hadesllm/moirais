"""Leave-one-out cross-validation for variogram models."""

from __future__ import annotations

from ._containers import DescriptiveResult


def cross_validation_variogram(Z, coords, model="spherical", params=None):
    """Leave-one-out cross-validation for a variogram/kriging model.

    .. epigraph:: "Hesitation is defeat." -- Isshin Ashina, Sekiro

    Parameters
    ----------
    Z : array_like
        Observed values.
    coords : array_like
        Coordinates, shape ``(n, 2)``.
    model : str
        Variogram model name.
    params : dict, optional
        ``{'nugget': ..., 'sill': ..., 'range': ...}``. Auto-fit if None.

    Returns
    -------
    DescriptiveResult
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    Z = np.asarray(Z, dtype=np.float64).ravel()
    coords = np.asarray(coords, dtype=np.float64)
    n = len(Z)

    D = squareform(pdist(coords))

    if params is None:
        nugget = 0.0
        sill = float(np.var(Z))
        range_p = float(D.max() / 3.0)
    else:
        nugget = params.get("nugget", 0.0)
        sill = params.get("sill", np.var(Z))
        range_p = params.get("range", D.max() / 3.0)

    def _gamma(h):
        if model == "spherical":
            return np.where(
                h <= range_p, nugget + (sill - nugget) * (1.5 * h / range_p - 0.5 * (h / range_p) ** 3), sill
            )
        elif model == "exponential":
            return nugget + (sill - nugget) * (1.0 - np.exp(-h / range_p))
        else:
            return nugget + (sill - nugget) * (1.0 - np.exp(-((h / range_p) ** 2)))

    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        d_i = D[i, mask]
        z_i = Z[mask]

        gamma_vals = _gamma(d_i)
        cov_vals = sill - gamma_vals
        w = cov_vals / (cov_vals.sum() + 1e-10)
        z_hat = np.sum(w * z_i)
        errors[i] = Z[i] - z_hat

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    return DescriptiveResult(
        name="cross_validation_variogram",
        value=rmse,
        extra={
            "rmse": rmse,
            "mae": mae,
            "mean_error": float(np.mean(errors)),
            "errors": errors.tolist(),
            "model": model,
        },
    )


sgcvg = cross_validation_variogram


def cheatsheet() -> str:
    return "cross_validation_variogram({}) -> Leave-one-out cross-validation for variogram models."
