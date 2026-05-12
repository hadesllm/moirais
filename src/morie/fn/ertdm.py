# morie.fn — function file (hadesllm/morie)
"""Earth mover's (Wasserstein) distance. 'This is the way.' -- The Mandalorian"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from ._containers import DescriptiveResult


def earth_movers_dist(
    p: np.ndarray, q: np.ndarray, p_weights: np.ndarray | None = None, q_weights: np.ndarray | None = None
) -> DescriptiveResult:
    r"""
    Compute the Earth Mover's Distance (1-Wasserstein) between two
    1-D distributions.

    .. math::

        W_1(P, Q) = \\int_0^1 |F_P^{-1}(t) - F_Q^{-1}(t)| \\, dt

    :param p: Values of the first distribution.
    :type p: numpy.ndarray
    :param q: Values of the second distribution.
    :type q: numpy.ndarray
    :param p_weights: Optional weights for p. Default uniform.
    :type p_weights: numpy.ndarray or None
    :param q_weights: Optional weights for q. Default uniform.
    :type q_weights: numpy.ndarray or None
    :return: DescriptiveResult with Wasserstein distance.
    :rtype: DescriptiveResult

    References
    ----------
    Rubner Y., Tomasi C. & Guibas L.J. (2000). The earth mover's
    distance as a metric for image retrieval. *International Journal of
    Computer Vision*, 40(2), 99-121.

    Vaserstein L.N. (1969). Markov processes over denumerable products of
    spaces, describing large systems of automata. *Problemy Peredachi
    Informatsii*, 5(3), 64-72.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    kwargs = {}
    if p_weights is not None:
        kwargs["u_weights"] = np.asarray(p_weights, dtype=float).ravel()
    if q_weights is not None:
        kwargs["v_weights"] = np.asarray(q_weights, dtype=float).ravel()
    dist = float(wasserstein_distance(p, q, **kwargs))
    return DescriptiveResult(
        name="earth_movers_distance",
        value=dist,
        extra={"distance": dist, "n_p": len(p), "n_q": len(q)},
    )


ertdm = earth_movers_dist


def cheatsheet() -> str:
    return "earth_movers_dist({}) -> Earth mover's (Wasserstein) distance. 'This is the way.' -- "
