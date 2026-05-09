"""Symmetrization via Rademacher complexity."""

from __future__ import annotations

import numpy as np


def symmetrization_bound(
    x: np.ndarray,
    *,
    fn: str = "mean",
    n_rademacher: int = 1000,
    seed: int = 42,
) -> dict:
    r"""
    Symmetrization technique bounding empirical process suprema via
    Rademacher averages.

    The symmetrization lemma (Gine & Zinn, 1984) states that for a
    class :math:`\mathcal{F}` of measurable functions:

    .. math::

        E\!\left[\sup_{f \in \mathcal{F}} \left|\frac{1}{n}
        \sum_{i=1}^n f(X_i) - Ef\right|\right]
        \le 2\, E\!\left[\sup_{f \in \mathcal{F}} \left|\frac{1}{n}
        \sum_{i=1}^n \sigma_i f(X_i)\right|\right]

    where :math:`\sigma_i` are iid Rademacher variables (:math:`\pm 1`
    with equal probability).

    This function estimates the Rademacher average empirically for a
    chosen summary functional applied to the data.

    :param x: 1-D array of observations.
    :param fn: Functional to evaluate. One of ``"mean"``, ``"var"``,
        ``"max"``, ``"abs_mean"``. Default ``"mean"``.
    :param n_rademacher: Number of Rademacher draws. Default 1000.
    :param seed: Random seed. Default 42.
    :return: dict with ``rademacher_avg`` (estimated Rademacher average),
        ``symmetrization_bound`` (2x Rademacher average),
        ``empirical_value``, ``n``.
    :raises ValueError: If x is empty or fn is unknown.

    References
    ----------
    Gine, E. & Zinn, J. (1984). Some limit theorems for empirical
        processes. *Ann. Probab.*, 12(4), 929--989.
    Kosorok, M.R. (2008). *Introduction to Empirical Processes and
        Semiparametric Inference*, Sec. 8.1. Springer.
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("x must be non-empty.")

    fns = {
        "mean": lambda v: np.mean(v),
        "var": lambda v: np.var(v, ddof=1) if len(v) > 1 else 0.0,
        "max": lambda v: np.max(np.abs(v)),
        "abs_mean": lambda v: np.mean(np.abs(v)),
    }
    if fn not in fns:
        raise ValueError(f"fn must be one of {set(fns)}, got '{fn}'.")

    emp_val = float(fns[fn](x))
    rng = np.random.default_rng(seed)
    n = x.size

    rad_vals = np.empty(n_rademacher)
    for k in range(n_rademacher):
        sigma = rng.choice([-1.0, 1.0], size=n)
        rad_vals[k] = fns[fn](sigma * x)

    rad_avg = float(np.mean(np.abs(rad_vals)))

    return {
        "rademacher_avg": rad_avg,
        "symmetrization_bound": 2.0 * rad_avg,
        "empirical_value": emp_val,
        "n": n,
        "n_rademacher": n_rademacher,
        "method": "Symmetrization (Rademacher)",
    }


symtz = symmetrization_bound


def cheatsheet() -> str:
    return "symmetrization_bound({x}) -> Symmetrization via Rademacher complexity."
