# moirais.fn — function file (hadesllm/moirais)
"""Frailty model — gamma-distributed shared frailty for survival."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def frailty_model(time, event, group, theta_init=1.0):
    """
    Gamma shared frailty model via marginal likelihood.

    Extends the Cox model with a random effect for group-level heterogeneity.

    :param time: (n,) survival times.
    :param event: (n,) event indicator (1=event, 0=censored).
    :param group: (n,) group membership.
    :param theta_init: Initial frailty variance.
    :return: DescriptiveResult with frailty variance, group frailties.

    References
    ----------
    Hougaard P (2000). Analysis of Multivariate Survival Data. Springer.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.int64).ravel()
    group = np.asarray(group).ravel()
    groups = np.unique(group)
    n = len(time)

    def _neg_marginal_ll(log_theta):
        theta = np.exp(log_theta)
        ll = 0.0
        for g in groups:
            mask = group == g
            d_g = event[mask].sum()
            sum_h0 = np.sum(time[mask])
            from scipy.special import gammaln

            ll += gammaln(1 / theta + d_g) - gammaln(1 / theta)
            ll += d_g * np.log(theta) if theta > 0 else 0
            ll -= (1 / theta + d_g) * np.log(1 + theta * sum_h0)
        return -ll

    res = optimize.minimize_scalar(_neg_marginal_ll, bounds=(-5, 5), method="bounded")
    theta_hat = np.exp(res.x)

    frailties = {}
    for g in groups:
        mask = group == g
        d_g = event[mask].sum()
        sum_h0 = np.sum(time[mask])
        frailties[str(g)] = (1 / theta_hat + d_g) / (1 / theta_hat + sum_h0)

    return DescriptiveResult(
        name="frailty_model",
        value=float(theta_hat),
        extra={
            "frailty_variance": float(theta_hat),
            "group_frailties": frailties,
            "n_groups": len(groups),
            "neg_log_lik": float(res.fun),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "frailty_model({}) -> Frailty model — gamma-distributed shared frailty for surviva"
