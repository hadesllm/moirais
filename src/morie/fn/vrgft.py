"""Variogram model fitting (spherical/exponential/Gaussian)."""
import numpy as np
from scipy import optimize
from scipy.spatial.distance import pdist, squareform
from ._richresult import RichResult

__all__ = ["variogram_fitting"]


def _empirical_variogram(x, coords, n_bins=10, max_dist=None):
    x = np.asarray(x, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    n = x.size
    D = squareform(pdist(coords))
    iu = np.triu_indices(n, k=1)
    dists = D[iu]
    diffs2 = (x[iu[0]] - x[iu[1]]) ** 2
    if max_dist is None:
        max_dist = float(dists.max() / 2.0)
    edges = np.linspace(0.0, max_dist, n_bins + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    gamma = np.full(n_bins, np.nan)
    npairs = np.zeros(n_bins, dtype=int)
    for k in range(n_bins):
        mask = (dists > edges[k]) & (dists <= edges[k + 1])
        m = int(mask.sum())
        npairs[k] = m
        if m > 0:
            gamma[k] = 0.5 * diffs2[mask].mean()
    keep = np.isfinite(gamma) & (npairs > 0)
    return mid[keep], gamma[keep], npairs[keep], n


def _model(h, c0, c1, a, kind):
    h = np.asarray(h, dtype=float)
    if kind == "exponential":
        return c0 + c1 * (1.0 - np.exp(-h / a))
    if kind == "gaussian":
        return c0 + c1 * (1.0 - np.exp(-(h ** 2) / (a ** 2)))
    if kind == "spherical":
        return np.where(
            h <= a,
            c0 + c1 * (1.5 * h / a - 0.5 * (h / a) ** 3),
            c0 + c1,
        )
    raise ValueError(f"unknown model: {kind}")


def variogram_fitting(x, coords, model: str = "exponential",
                      n_bins: int = 10, max_dist: float | None = None):
    """
    Variogram model fitting by weighted least squares on the empirical
    semivariogram (Cressie weights ~ n_pairs / gamma^2).

    Models supported:  ``exponential``, ``gaussian``, ``spherical``.

    Exponential form:  gamma(h) = c0 + c1 * (1 - exp(-h / a))

    Parameters
    ----------
    x : array-like, shape (n,)
    coords : array-like, shape (n, d)
    model : str
    n_bins : int
    max_dist : float, optional

    Returns
    -------
    RichResult with payload:
        estimate : {model, nugget=c0, sill=c0+c1, range=a, params=(c0,c1,a)}
        n, method
    """
    mids, gammas, npairs, n = _empirical_variogram(x, coords, n_bins, max_dist)
    # Auto-shrink bins if too few were populated
    while mids.size < 3 and n_bins > 3:
        n_bins -= 1
        mids, gammas, npairs, n = _empirical_variogram(x, coords, n_bins, max_dist)
    if mids.size < 3:
        raise ValueError("need at least 3 non-empty bins")
    g_max = float(gammas.max())
    h_max = float(mids.max())
    p0 = [0.0, g_max, max(h_max / 3.0, 1e-6)]
    bounds = ([0.0, 1e-12, 1e-12], [g_max * 5 + 1e-6, g_max * 10 + 1.0, h_max * 10])
    weights = np.maximum(npairs.astype(float), 1.0) / np.maximum(gammas, 1e-12) ** 2

    def resid(p):
        c0, c1, a = p
        pred = _model(mids, c0, c1, a, model)
        return np.sqrt(weights) * (gammas - pred)

    try:
        res = optimize.least_squares(resid, p0, bounds=bounds, max_nfev=2000)
        c0, c1, a = (float(v) for v in res.x)
        converged = bool(res.success)
    except Exception:
        c0, c1, a = p0
        converged = False

    estimate = {
        "model": model,
        "nugget": c0,
        "sill": c0 + c1,
        "range": a,
        "params": [c0, c1, a],
        "converged": converged,
    }
    return RichResult(payload={
        "estimate": estimate,
        "n": int(n),
        "method": f"Variogram model fit ({model}, WLS)",
    })


def cheatsheet():
    return "vrgft: Variogram model fit (exponential/Gaussian/spherical)"


# CANONICAL TEST
# x=[1,2,3,4,5], coords=[[0],[1],[2],[3],[4]], model='exponential'
# Expect:  converged True, range > 0, sill > 0, nugget >= 0
