"""Slepian-Wolf bound for distributed source coding."""

__all__ = ["slepw"]

import numpy as np


def slepw(joint_pmf: np.ndarray) -> dict:
    """
    Compute the Slepian-Wolf rate region for two correlated sources.

    For jointly distributed (X, Y) with joint PMF p(x, y), the achievable
    rate region is:

    .. math::

        R_X \\ge H(X|Y), \\quad R_Y \\ge H(Y|X), \\quad R_X + R_Y \\ge H(X, Y)

    Parameters
    ----------
    joint_pmf : np.ndarray
        Joint probability table p(x, y), shape (n_x, n_y). Must sum to 1.

    Returns
    -------
    dict
        'H_X' (marginal entropy of X, bits),
        'H_Y' (marginal entropy of Y, bits),
        'H_XY' (joint entropy, bits),
        'H_X_given_Y' (conditional entropy H(X|Y), bits),
        'H_Y_given_X' (conditional entropy H(Y|X), bits),
        'R_X_min' (minimum rate for X),
        'R_Y_min' (minimum rate for Y),
        'R_sum_min' (minimum sum rate).

    Raises
    ------
    ValueError
        If joint_pmf is not 2-D or does not sum to 1.

    References
    ----------
    Slepian, D. & Wolf, J. K. (1973). Noiseless coding of correlated
    information sources. IEEE Trans. Inform. Theory, 19(4), 471-480.
    """
    pxy = np.asarray(joint_pmf, dtype=np.float64)
    if pxy.ndim != 2:
        raise ValueError("joint_pmf must be 2-D.")
    if not np.isclose(pxy.sum(), 1.0):
        raise ValueError("joint_pmf must sum to 1.")
    if np.any(pxy < 0):
        raise ValueError("joint_pmf entries must be non-negative.")

    eps = 1e-300
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    h_x = -np.sum(px[px > eps] * np.log2(px[px > eps]))
    h_y = -np.sum(py[py > eps] * np.log2(py[py > eps]))

    flat = pxy.ravel()
    h_xy = -np.sum(flat[flat > eps] * np.log2(flat[flat > eps]))

    h_x_given_y = h_xy - h_y
    h_y_given_x = h_xy - h_x

    return {
        "H_X": h_x,
        "H_Y": h_y,
        "H_XY": h_xy,
        "H_X_given_Y": h_x_given_y,
        "H_Y_given_X": h_y_given_x,
        "R_X_min": h_x_given_y,
        "R_Y_min": h_y_given_x,
        "R_sum_min": h_xy,
    }
