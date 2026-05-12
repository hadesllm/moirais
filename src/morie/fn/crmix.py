# morie.fn -- function file (hadesllm/morie)
"""Cure rate mixture model -- long-term survivors."""

import numpy as np
from scipy import optimize

from ._containers import DescriptiveResult


def cure_rate_model(time, event):
    """
    Mixture cure model assuming a fraction of subjects are 'cured'.

    S(t) = pi + (1-pi) * S_u(t), where pi is the cure fraction and
    S_u(t) is the uncured survival (exponential).

    :param time: (n,) survival times.
    :param event: (n,) event indicator (1=event, 0=censored).
    :return: DescriptiveResult with cure fraction, hazard rate.

    References
    ----------
    Berkson J, Gage RP (1952). Survival Curve for Cancer Patients
    Following Treatment. JASA 47(259):501-515.
    """
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.int64).ravel()
    n = len(time)

    def _neg_ll(params):
        logit_pi, log_lam = params
        pi = 1 / (1 + np.exp(-logit_pi))
        lam = np.exp(log_lam)
        ll = 0.0
        for i in range(n):
            s_u = np.exp(-lam * time[i])
            if event[i] == 1:
                ll += np.log(max((1 - pi) * lam * s_u, 1e-300))
            else:
                ll += np.log(max(pi + (1 - pi) * s_u, 1e-300))
        return -ll

    res = optimize.minimize(_neg_ll, [0.0, -1.0], method="Nelder-Mead")
    logit_pi, log_lam = res.x
    pi = 1 / (1 + np.exp(-logit_pi))
    lam = np.exp(log_lam)

    return DescriptiveResult(
        name="cure_rate_model",
        value=float(pi),
        extra={
            "cure_fraction": float(pi),
            "hazard_rate": float(lam),
            "neg_log_lik": float(res.fun),
            "n_events": int(event.sum()),
            "n": n,
        },
    )


def cheatsheet() -> str:
    return "cure_rate_model({}) -> Cure rate mixture model -- long-term survivors."
