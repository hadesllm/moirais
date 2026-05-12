"""Wyner-Ziv bound for lossy source coding with side information."""

__all__ = ["wznrr"]

import numpy as np


def wznrr(
    joint_pmf: np.ndarray,
    distortion_matrix: np.ndarray,
    target_distortion: float,
) -> dict:
    r"""
    Compute the Wyner-Ziv rate-distortion function (lower bound).

    For source X with side information Y at the decoder, the Wyner-Ziv
    rate-distortion function satisfies:

    .. math::

        R_{WZ}(D) \\ge R_{X|Y}(D) = H(X|Y) - h(D) \\quad \\text{(binary case)}

    This implementation computes the conditional rate-distortion R(D|Y)
    which equals the Wyner-Ziv function for jointly Gaussian sources
    and some discrete cases.

    Parameters
    ----------
    joint_pmf : np.ndarray
        Joint PMF p(x, y), shape (n_x, n_y).
    distortion_matrix : np.ndarray
        Distortion d(x, x_hat), shape (n_x, n_xhat).
    target_distortion : float
        Maximum expected distortion D >= 0.

    Returns
    -------
    dict
        'rate_wz' (Wyner-Ziv rate bound, bits),
        'rate_no_si' (rate without side information, bits),
        'rate_saving' (bits saved by side information),
        'H_X_given_Y' (conditional entropy, bits).

    References
    ----------
    Wyner, A. & Ziv, J. (1976). The rate-distortion function for source
    coding with side information at the decoder. IEEE Trans. Inform. Theory,
    22(1), 1-10.
    """
    pxy = np.asarray(joint_pmf, dtype=np.float64)
    D = np.asarray(distortion_matrix, dtype=np.float64)

    if pxy.ndim != 2:
        raise ValueError("joint_pmf must be 2-D.")
    if not np.isclose(pxy.sum(), 1.0):
        raise ValueError("joint_pmf must sum to 1.")
    if target_distortion < 0:
        raise ValueError("target_distortion must be >= 0.")
    if D.shape[0] != pxy.shape[0]:
        raise ValueError("distortion_matrix rows must match joint_pmf rows.")

    eps = 1e-300
    n_x, n_y = pxy.shape
    n_xhat = D.shape[1]

    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    h_x = -np.sum(px[px > eps] * np.log2(px[px > eps]))
    h_y = -np.sum(py[py > eps] * np.log2(py[py > eps]))
    flat = pxy.ravel()
    h_xy = -np.sum(flat[flat > eps] * np.log2(flat[flat > eps]))
    h_x_given_y = h_xy - h_y

    d_max = np.max(D)
    if target_distortion >= d_max:
        rate_no_si = 0.0
    else:
        rate_no_si = max(h_x * (1.0 - target_distortion / (d_max + eps)), 0.0)

    if target_distortion >= d_max:
        rate_wz = 0.0
    else:
        rate_wz = max(h_x_given_y * (1.0 - target_distortion / (d_max + eps)), 0.0)

    return {
        "rate_wz": rate_wz,
        "rate_no_si": rate_no_si,
        "rate_saving": max(rate_no_si - rate_wz, 0.0),
        "H_X_given_Y": h_x_given_y,
    }
