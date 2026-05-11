# morie.fn — function file (hadesllm/morie)
"""Horvitz-Thompson estimator of population total."""

import math

import numpy as np
import scipy.stats as scipy_stats


def horvitz_thompson_total(
    y: np.ndarray,
    inclusion_probs: np.ndarray,
) -> dict:
    """
    Horvitz-Thompson estimator of population total.

    The HT estimator is:

    .. math::

        \\hat{\\tau}_{HT} = \\sum_{i \\in s} \\frac{y_i}{\\pi_i}

    where :math:`\\pi_i` is the first-order inclusion probability for unit i.

    The variance estimator used here is the Sen-Yates-Grundy (SYG) approximation
    assuming simple random sampling (i.e., pairwise inclusion probabilities
    :math:`\\pi_{ij} \\approx \\pi_i \\pi_j`), which yields:

    .. math::

        \\hat{V}(\\hat{\\tau}_{HT}) \\approx \\sum_i \\frac{y_i^2}{\\pi_i^2}
        \\cdot \\frac{1 - \\pi_i}{1}

    This is the standard Horvitz-Thompson variance under with-replacement
    sampling (Hansen-Hurwitz estimator) and is conservative under
    without-replacement designs.

    :param y: Observed response values for sampled units (1-D array-like).
    :param inclusion_probs: First-order inclusion probabilities pi_i for each
        sampled unit (values in (0, 1]).
    :return: dict with keys ``total``, ``se``, ``ci_lower``, ``ci_upper``.
    :raises ValueError: If y and inclusion_probs have different lengths, or
        any inclusion probability is <= 0 or > 1.

    References
    ----------
    Horvitz, D. G., & Thompson, D. J. (1952). A generalisation of sampling
        without replacement from a finite universe. JASA, 47(260), 663-685.
    Cochran, W. G. (1977). Sampling Techniques (3rd ed.). Wiley. (Chapter 9A.)
    """
    y_arr = np.asarray(y, dtype=float)
    pi_arr = np.asarray(inclusion_probs, dtype=float)
    if len(y_arr) != len(pi_arr):
        raise ValueError(f"y and inclusion_probs must have the same length; got {len(y_arr)} and {len(pi_arr)}.")
    if np.any(pi_arr <= 0) or np.any(pi_arr > 1):
        raise ValueError("All inclusion_probs must be in (0, 1].")
    if len(y_arr) == 0:
        raise ValueError("y must contain at least one observation.")

    # HT total: sum of y_i / pi_i
    ht_total = float(np.sum(y_arr / pi_arr))

    # Variance approximation: SYG with pi_ij approx pi_i * pi_j (with-replacement approx)
    # V_HT approx sum_i (y_i / pi_i)^2 * (1 - pi_i)
    z = y_arr / pi_arr  # expansion values
    var_ht = float(np.sum(z**2 * (1.0 - pi_arr)))
    se = math.sqrt(max(0.0, var_ht))

    # Approximate 95% CI using normal quantile (valid for large samples)
    z_crit = float(scipy_stats.norm.ppf(0.975))
    return {
        "total": ht_total,
        "se": se,
        "ci_lower": ht_total - z_crit * se,
        "ci_upper": ht_total + z_crit * se,
    }


ht_tot_fn = horvitz_thompson_total


def cheatsheet() -> str:
    return "horvitz_thompson_total({}) -> Horvitz-Thompson estimator of population total."
