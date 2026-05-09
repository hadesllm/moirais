# moirais.fn — function file (hadesllm/moirais)
"""Earth mover's (Wasserstein-1) distance."""

import numpy as np

from ._containers import ESRes
def earth_mover_dist(p, q, **kwargs) -> ESRes:
    """
    Compute the Earth Mover's Distance (Wasserstein-1) between two
    1-D distributions.

    Uses the closed-form for 1-D: EMD = integral |CDF_P - CDF_Q|.

    :param p: array-like, first distribution (counts or probabilities).
    :param q: array-like, second distribution (counts or probabilities).
    :return: ESRes with EMD value.

    References
    ----------
    Rubner Y, Tomasi C, Guibas LJ (2000). The earth mover's
    distance as a metric for image retrieval. IJCV, 40(2), 99-121.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if len(p) != len(q):
        raise ValueError("p and q must have same length.")
    p = p / p.sum()
    q = q / q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    emd = float(np.sum(np.abs(cdf_p - cdf_q)))
    return ESRes(measure="earth_mover_distance", estimate=emd, extra={"n_categories": len(p)})


emdd = earth_mover_dist


def cheatsheet() -> str:
    return "earth_mover_dist({}) -> Earth mover's (Wasserstein-1) distance."
