# moirais.fn — function file (hadesllm/moirais)
"""Joint entropy."""

import numpy as np

from ._containers import ESRes

_QUOTE = "The belonging you seek is not behind you, it is ahead. -- Maz Kanata"


def joint_entropy(x, y, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute joint Shannon entropy H(X, Y).

    .. math::

        H(X,Y) = -\\sum_{i,j} p(x_i, y_j) \\log_2 p(x_i, y_j)

    :param x: array-like, first variable.
    :param y: array-like, second variable.
    :param bins: Number of bins per dimension.
    :return: ESRes with joint entropy in bits.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    hist, _, _ = np.histogram2d(x, y, bins=bins)
    p = hist / hist.sum()
    p = p[p > 0]
    h = -float(np.sum(p * np.log2(p)))
    return ESRes(
        measure="joint_entropy",
        estimate=h,
        n=len(x),
        extra={"bits": h, "n_bins": bins},
    )


jnent = joint_entropy


def cheatsheet() -> str:
    return "joint_entropy({}) -> Joint entropy."
