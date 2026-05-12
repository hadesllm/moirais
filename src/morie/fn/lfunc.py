# morie.fn — function file (hadesllm/morie)
"""Ripley's L function for spatial point patterns."""

import numpy as np

from ._containers import DescriptiveResult
from .kfunc import ripley_k


def ripley_l(points: np.ndarray, distances: np.ndarray | None = None, n_distances: int = 20) -> DescriptiveResult:
    r"""
    Compute Ripley's L function.

    .. math::

        L(d) = \\sqrt{K(d) / \\pi} - d

    Under CSR, L(d) = 0 for all d. L > 0 indicates clustering.

    :param points: (n, 2) array of coordinates.
    :param distances: Distances at which to evaluate L (optional).
    :param n_distances: Number of evenly spaced distances if not provided.
    :return: DescriptiveResult with L values and distances.

    References
    ----------
    Besag J (1977). Contribution to the discussion of Dr Ripley's paper.
    Journal of the Royal Statistical Society B, 39, 193-195.
    """
    k_res = ripley_k(points, distances, n_distances)
    K_vals = k_res.extra["K"]
    dists = k_res.extra["distances"]
    L_vals = np.sqrt(np.maximum(K_vals, 0) / np.pi) - dists
    return DescriptiveResult(
        name="ripley_l",
        value=float(L_vals[-1]) if len(L_vals) > 0 else 0.0,
        extra={"L": L_vals, "distances": dists, "K": K_vals, "n_points": k_res.extra["n_points"]},
    )


lfunc = ripley_l


def cheatsheet() -> str:
    return "ripley_l({}) -> Ripley's L function for spatial point patterns."
