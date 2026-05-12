# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Causal bounds -- partial identification under unmeasured confounding."""

import numpy as np

from ._containers import DescriptiveResult


def causal_bounds(y, treatment, p_upper=None, p_lower=None):
    """
    Compute Manski-type causal bounds for ATE under partial identification.

    When unmeasured confounding prevents point identification, these bounds
    bracket the true ATE.

    :param y: (n,) outcome (must be bounded in [0, 1] or will be rescaled).
    :param treatment: (n,) binary treatment indicator.
    :param p_upper: Upper bound on Y. Defaults to max(y).
    :param p_lower: Lower bound on Y. Defaults to min(y).
    :return: DescriptiveResult with ATE lower/upper bounds.

    References
    ----------
    Manski CF (1990). Nonparametric Bounds on Treatment Effects.
    American Economic Review 80(2):319-323.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    t = np.asarray(treatment, dtype=np.float64).ravel()
    if p_upper is None:
        p_upper = float(np.max(y))
    if p_lower is None:
        p_lower = float(np.min(y))

    t1 = t == 1
    t0 = t == 0
    n1, n0 = t1.sum(), t0.sum()
    if n1 == 0 or n0 == 0:
        raise ValueError("Need observations in both treatment groups")

    e1 = y[t1].mean()
    e0 = y[t0].mean()
    p = n1 / len(y)

    lb = (p * e1 + (1 - p) * p_lower) - (p * p_upper + (1 - p) * e0)
    ub = (p * e1 + (1 - p) * p_upper) - (p * p_lower + (1 - p) * e0)

    return DescriptiveResult(
        name="causal_bounds",
        value=float((lb + ub) / 2),
        extra={
            "ate_lower": float(lb),
            "ate_upper": float(ub),
            "ate_midpoint": float((lb + ub) / 2),
            "naive_ate": float(e1 - e0),
            "prop_treated": float(p),
            "n_treated": int(n1),
            "n_control": int(n0),
        },
    )


def cheatsheet() -> str:
    return "causal_bounds({}) -> Causal bounds -- partial identification under unmeasured conf"
