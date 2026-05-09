# moirais.fn — function file (hadesllm/moirais)
"""Survey ratio estimator for a population total."""

import math

import numpy as np
import scipy.stats as scipy_stats


def ratio_estimator(
    y: np.ndarray,
    x: np.ndarray,
    weights: np.ndarray,
    X_population_total: float,
) -> dict:
    """
    Survey ratio estimator for a population total.

    The ratio estimator exploits auxiliary information X whose population
    total is known:

    .. math::

        \\hat{Y}_R = \\hat{R} \\cdot X_{\\text{pop}}

    where :math:`\\hat{R} = \\hat{Y}_{HT} / \\hat{X}_{HT}` is the estimated
    ratio of totals.

    Standard error via Taylor linearisation:

    .. math::

        \\hat{V}(\\hat{Y}_R) \\approx \\frac{1}{\\hat{X}_{HT}^2}
        \\sum_i w_i^2 (y_i - \\hat{R} x_i)^2

    :param y: Response variable values for sampled units (1-D array-like).
    :param x: Auxiliary variable values for sampled units (same length as y).
    :param weights: Survey weights w_i (must be > 0).
    :param X_population_total: Known population total of the auxiliary variable.
    :return: dict with keys ``ratio``, ``total_estimate``, ``se``,
        ``ci_lower``, ``ci_upper``.
    :raises ValueError: If inputs have different lengths, weights <= 0, or
        X_population_total <= 0.

    References
    ----------
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Section 6.4.)
    Lumley, T. (2010). Complex Surveys: A Guide to Analysis Using R. Wiley.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    if len(y_arr) != len(x_arr) or len(y_arr) != len(w_arr):
        raise ValueError("y, x, and weights must all have the same length.")
    if np.any(w_arr <= 0):
        raise ValueError("All weights must be > 0.")
    if X_population_total <= 0:
        raise ValueError(f"X_population_total must be > 0, got {X_population_total}.")

    y_ht = float(np.sum(w_arr * y_arr))
    x_ht = float(np.sum(w_arr * x_arr))

    if x_ht == 0:
        raise ValueError("Weighted sum of auxiliary variable x is zero.")

    r_hat = y_ht / x_ht
    total_estimate = r_hat * X_population_total

    # Taylor linearisation SE
    residuals = y_arr - r_hat * x_arr
    var_est = float(np.sum(w_arr**2 * residuals**2)) / (x_ht**2) * (X_population_total**2)
    se = math.sqrt(max(0.0, var_est))

    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "ratio": float(r_hat),
        "total_estimate": float(total_estimate),
        "se": float(se),
        "ci_lower": float(total_estimate - z_crit * se),
        "ci_upper": float(total_estimate + z_crit * se),
    }


ratio_fn = ratio_estimator


def cheatsheet() -> str:
    return "ratio_estimator({}) -> Survey ratio estimator for a population total."
