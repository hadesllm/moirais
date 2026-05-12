# morie.fn — function file (hadesllm/morie)
"""Mutual information between two discrete variables."""

import numpy as np

from ._containers import ESRes


def mutual_information(x: np.ndarray, y: np.ndarray, base: float = 2.0) -> ESRes:
    r"""
    Compute mutual information between two discrete variables.

    .. math::

        I(X;Y) = \\sum_{x,y} p(x,y) \\log \\frac{p(x,y)}{p(x)p(y)}

    :param x: (n,) discrete variable.
    :param y: (n,) discrete variable.
    :param base: Logarithm base.
    :return: ESRes with MI value.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory.
    2nd ed. Wiley.
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have same length.")
    ux, ix = np.unique(x, return_inverse=True)
    uy, iy = np.unique(y, return_inverse=True)
    joint = np.zeros((len(ux), len(uy)))
    for i in range(n):
        joint[ix[i], iy[i]] += 1
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(len(ux)):
        for j in range(len(uy)):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j])) / np.log(base)
    return ESRes(measure="mutual_information", estimate=float(mi), n=n, extra={"base": base})


mutl = mutual_information


def cheatsheet() -> str:
    return "mutual_information({}) -> Mutual information between two discrete variables."
