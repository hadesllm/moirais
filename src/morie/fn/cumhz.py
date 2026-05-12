# morie.fn -- function file (hadesllm/morie)
"""Cumulative hazard from Kaplan-Meier survival estimates."""

import numpy as np


def cumhz(
    times: np.ndarray,
    survival: np.ndarray,
) -> dict:
    r"""
    Cumulative hazard function derived from survival probabilities.

    Converts a Kaplan-Meier (or any) survival curve to the cumulative hazard
    via the identity:

    .. math::

        H(t) = -\\log S(t)

    :param times: 1-D array of event times (from a KM or similar estimator).
    :param survival: 1-D array of survival probabilities :math:`S(t)` at
        corresponding times.  Values must be in (0, 1].
    :return: dict with ``times`` and ``cumulative_hazard`` arrays.
    :raises ValueError: If arrays differ in length, are empty, or survival
        contains values outside (0, 1].

    References
    ----------
    Klein, J. P. & Moeschberger, M. L. (2003). Survival Analysis:
    Techniques for Censored and Truncated Data (2nd ed.). Springer.
    """
    times = np.asarray(times, dtype=float)
    survival = np.asarray(survival, dtype=float)
    if len(times) != len(survival):
        raise ValueError(f"times and survival must have same length, got {len(times)} and {len(survival)}.")
    if len(times) == 0:
        raise ValueError("Input arrays must not be empty.")
    if np.any(survival <= 0) or np.any(survival > 1):
        raise ValueError("survival values must be in (0, 1].")

    cum_hazard = -np.log(survival)
    return {
        "times": times,
        "cumulative_hazard": cum_hazard,
    }


def cheatsheet() -> str:
    return "cumhz({}) -> Cumulative hazard from Kaplan-Meier survival estimates."
