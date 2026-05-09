# moirais.fn — function file (hadesllm/moirais)
"""Conditional entropy."""

import numpy as np

from ._containers import ESRes
def conditional_entropy(x, y, bins: int = 50, **kwargs) -> ESRes:
    """
    Compute conditional entropy H(X|Y) = H(X,Y) - H(Y).

    .. math::

        H(X|Y) = H(X,Y) - H(Y)

    :param x: array-like, target variable.
    :param y: array-like, conditioning variable.
    :param bins: Number of bins per dimension.
    :return: ESRes with conditional entropy in bits.

    References
    ----------
    Cover TM, Thomas JA (2006). Elements of Information Theory,
    2nd ed. Wiley, New York.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have same length.")
    hist_joint, _, _ = np.histogram2d(x, y, bins=bins)
    pj = hist_joint / hist_joint.sum()
    pj_pos = pj[pj > 0]
    h_xy = -float(np.sum(pj_pos * np.log2(pj_pos)))
    hist_y, _ = np.histogram(y, bins=bins)
    py = hist_y / hist_y.sum()
    py_pos = py[py > 0]
    h_y = -float(np.sum(py_pos * np.log2(py_pos)))
    h_x_given_y = h_xy - h_y
    return ESRes(
        measure="conditional_entropy",
        estimate=h_x_given_y,
        n=len(x),
        extra={"h_xy": h_xy, "h_y": h_y, "bits": h_x_given_y},
    )


cdent = conditional_entropy


def cheatsheet() -> str:
    return "conditional_entropy({}) -> Conditional entropy."
