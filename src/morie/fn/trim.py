"""Propensity score trimming for overlap enforcement."""

from __future__ import annotations

from typing import Union

import numpy as np


def ps_trim(
    ps_scores: Union[list, np.ndarray],
    *,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Trim observations with extreme propensity scores.

    Returns a boolean mask where True indicates the observation should
    be *kept* (i.e., its propensity score is in [threshold, 1 - threshold]).

    Common thresholds: 0.1 (Crump et al., 2009), 0.05, 0.01.

    :param ps_scores: Estimated propensity scores (1-D array, values in [0, 1]).
    :param threshold: Trimming threshold (default 0.1). Observations with
        PS < threshold or PS > 1 - threshold are flagged for removal.
    :return: Boolean array (True = keep).
    :raises ValueError: If threshold not in (0, 0.5).

    References
    ----------
    Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009).
    Dealing with limited overlap in estimation of average treatment
    effects. *Biometrika*, 96(1), 187--199.
    """
    if not (0 < threshold < 0.5):
        raise ValueError("threshold must be in (0, 0.5).")
    ps = np.asarray(ps_scores, dtype=float)
    return (ps >= threshold) & (ps <= 1.0 - threshold)


trim = ps_trim


def cheatsheet() -> str:
    return "ps_trim({}) -> Propensity score trimming for overlap enforcement."
