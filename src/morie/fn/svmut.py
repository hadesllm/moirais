"""Mixed (quadratic+Gaussian) utility."""

import numpy as np

from ._containers import SpatialResult


def svmut(ideal, pos, *, alpha=0.5, bandwidth=1.0):
    """Mixed (quadratic+Gaussian) utility.

    Returns
    -------
    SpatialResult
    """

    ideal = np.asarray(ideal, dtype=float)
    pos = np.asarray(pos, dtype=float)
    diff = ideal - pos
    quad = -float(np.dot(diff, diff))
    gauss = float(np.exp(-np.dot(diff, diff) / (2 * bandwidth**2)))
    stat = alpha * quad + (1 - alpha) * gauss
    return SpatialResult(
        name="Mixed Quadratic-Gaussian Utility",
        statistic=float(stat),
        extra={"quadratic": quad, "gaussian": gauss, "alpha": alpha},
    )


svmut = svmut  # alias


def cheatsheet() -> str:
    return "svmut({}) -> Mixed (quadratic+Gaussian) utility."
