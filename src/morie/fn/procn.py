# morie.fn — function file (hadesllm/morie)
"""Process convolution model (Schabenberger & Gotway Ch 8)."""

import numpy as np
from scipy.spatial.distance import cdist


def procn(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    n_knots: int = 10,
    bandwidth: float | None = None,
    seed: int | None = None,
) -> dict:
    """
    Fit a process convolution (kernel convolution) spatial model.

    The spatial process is modelled as a convolution of a white-noise
    process with a smoothing kernel centred at a set of knots, yielding
    a valid non-stationary covariance (Higdon 1998).

    :param coords: Observation coordinates (n, 2).
    :param values: Observed values (n,).
    :param n_knots: Number of knots for the latent process.
    :param bandwidth: Kernel bandwidth; defaults to median pairwise distance.
    :param seed: Random seed for knot placement.
    :return: dict with ``fitted``, ``knots``, ``weights``, ``bandwidth``.
    :raises ValueError: If shapes are incompatible.

    References
    ----------
    Higdon, D. (1998). A process-convolution approach to modelling
    temperatures in the North Atlantic Ocean. *Environmental and
    Ecological Statistics*, 5(2), 173-190.

    Schabenberger & Gotway (2005), Ch. 8.
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = int(values) if values.ndim == 0 else len(values)
    if coords.shape != (n, 2):
        raise ValueError(f"coords must be ({n}, 2), got {coords.shape}.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_knots, n), replace=False)
    knots = coords[idx]

    dists = cdist(coords, knots)
    if bandwidth is None:
        all_d = cdist(coords, coords)
        bandwidth = float(np.median(all_d[all_d > 0]))
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    K = np.exp(-0.5 * (dists / bandwidth) ** 2)

    weights, _, _, _ = np.linalg.lstsq(K, values, rcond=None)
    fitted = K @ weights

    return {
        "fitted": fitted,
        "knots": knots,
        "weights": weights,
        "bandwidth": bandwidth,
        "residuals": values - fitted,
        "n": n,
    }


procn_fn = procn


def cheatsheet() -> str:
    return "procn({}) -> Process convolution spatial model."
