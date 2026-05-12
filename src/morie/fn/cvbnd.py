# morie.fn -- function file (hadesllm/morie)
"""Cross-validation risk bound."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["cvbnd"]


def cvbnd(
    cv_risks: np.ndarray,
    n: int,
    *,
    n_folds: int = 5,
    delta: float = 0.05,
) -> dict[str, Any]:
    r"""
    Compute finite-sample cross-validation risk bounds.

    Uses the Hoeffding-type inequality for cross-validation:

    .. math::

        P\!\left(|R_{cv}(\hat{f}) - R(\hat{f})| > t\right)
        \le 2\exp\!\left(-\frac{2nt^2}{B^2}\right)

    :param cv_risks: Array of per-fold risk estimates, shape (n_folds,).
    :param n: Total sample size.
    :param n_folds: Number of CV folds. Default 5.
    :param delta: Confidence parameter. Default 0.05.
    :return: Dict with ``cv_risk``, ``cv_se``, ``upper_bound``,
        ``bound_width``, ``n``, ``delta``.
    :raises ValueError: If cv_risks is empty.

    References
    ----------
    Kosorok, M.R. (2008). Ch. 22. Springer.
    """
    cv_risks = np.asarray(cv_risks, dtype=float).ravel()
    if cv_risks.size == 0:
        raise ValueError("cv_risks must be non-empty.")

    cv_risk = float(np.mean(cv_risks))
    cv_se = float(np.std(cv_risks, ddof=1) / np.sqrt(len(cv_risks)))

    risk_range = float(np.max(cv_risks) - np.min(cv_risks)) if len(cv_risks) > 1 else 1.0
    bound_width = risk_range * np.sqrt(np.log(2.0 / delta) / (2.0 * n))
    upper_bound = cv_risk + bound_width

    return {
        "cv_risk": cv_risk,
        "cv_se": cv_se,
        "upper_bound": float(upper_bound),
        "bound_width": float(bound_width),
        "n": n,
        "delta": delta,
    }


def cheatsheet() -> str:
    return "cvbnd({cv_risks, n}) -> Cross-validation risk bound."
