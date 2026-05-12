# morie.fn -- function file (hadesllm/morie)
"""Hajek estimator for population mean."""

import math

import numpy as np
import scipy.stats as scipy_stats


def hajek_mean(
    y: np.ndarray,
    weights: np.ndarray,
) -> dict:
    r"""
    Hajek estimator for population mean (ratio estimator).

    The Hajek estimator normalises the HT estimator by the estimated
    population size, eliminating sensitivity to total weight magnitude:

    .. math::

        \\bar{y}_H = \\frac{\\sum_i w_i y_i}{\\sum_i w_i}

    where :math:`w_i = 1 / \\pi_i` are the survey weights.

    Standard error is computed via Taylor (delta-method) linearisation:

    .. math::

        \\hat{V}(\\bar{y}_H) \\approx \\frac{1}{N^2} \\sum_i w_i^2 (y_i - \\bar{y}_H)^2
        \\cdot \\frac{1 - \\pi_i}{\\pi_i}

    For the with-replacement approximation (pi_i = w_i / N_hat) this simplifies
    to the weighted sample variance divided by the effective sample size.

    :param y: Response values (1-D array-like).
    :param weights: Survey weights w_i = 1/pi_i (must be > 0).
    :return: dict with keys ``mean``, ``se``, ``ci_lower``, ``ci_upper``.
    :raises ValueError: If y and weights have different lengths or weights <= 0.

    References
    ----------
    Hajek, J. (1971). Comment in Godambe & Sprott (Eds.), Foundations of
        Statistical Inference. Holt, Rinehart and Winston.
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Section 6.13.)
    """
    y_arr = np.asarray(y, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if len(y_arr) != len(w_arr):
        raise ValueError(f"y and weights must have the same length; got {len(y_arr)} and {len(w_arr)}.")
    if np.any(w_arr <= 0):
        raise ValueError("All weights must be > 0.")
    if len(y_arr) < 2:
        raise ValueError("At least 2 observations are required for SE computation.")

    sum_w = float(np.sum(w_arr))
    hajek_mean_val = float(np.sum(w_arr * y_arr) / sum_w)

    # Taylor linearisation SE: weighted variance of residuals scaled by N_hat^2
    residuals = y_arr - hajek_mean_val
    # With-replacement approximation: treat pi_i approx w_i / sum_w => (1 - pi_i) approx 1
    # V(y_bar_H) approx (1/sum_w^2) * sum(w_i^2 * (y_i - y_bar_H)^2)
    var_hajek = float(np.sum(w_arr**2 * residuals**2)) / (sum_w**2)
    se = math.sqrt(max(0.0, var_hajek))

    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "mean": hajek_mean_val,
        "se": se,
        "ci_lower": hajek_mean_val - z_crit * se,
        "ci_upper": hajek_mean_val + z_crit * se,
    }


h_mean_fn = hajek_mean


def cheatsheet() -> str:
    return "hajek_mean({}) -> Hajek estimator for population mean."
