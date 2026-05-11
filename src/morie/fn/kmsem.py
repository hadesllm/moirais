# morie.fn — function file (hadesllm/morie)
"""Kaplan-Meier semiparametric survival estimator."""

from __future__ import annotations

import numpy as np


def kmsem(
    time: np.ndarray,
    event: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    r"""
    Kaplan-Meier product-limit estimator with Greenwood variance.

    The nonparametric maximum-likelihood estimator of the survival
    function from right-censored data:

    .. math::

        \hat{S}(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)

    where :math:`d_j` is the number of events at time :math:`t_j` and
    :math:`n_j` is the number at risk just prior.

    Greenwood's formula gives the variance:

    .. math::

        \widehat{\mathrm{Var}}[\hat{S}(t)]
        = \hat{S}(t)^2 \sum_{t_j \le t}
          \frac{d_j}{n_j (n_j - d_j)}

    Pointwise confidence intervals use the log-log transformation
    (Kalbfleisch & Prentice, 2002) for better boundary behavior.

    :param time: Observed times, shape ``(n,)``.
    :param event: Event indicator, shape ``(n,)``.
        1 = event, 0 = censored.
    :param alpha: Significance level for CIs. Default 0.05.
    :return: dict with ``times``, ``survival``, ``se``, ``ci_lower``,
        ``ci_upper``, ``n_obs``, ``n_events``, ``median_survival``.
    :raises ValueError: If inputs have different lengths or are empty.

    References
    ----------
    Kaplan, E. L. & Meier, P. (1958). Nonparametric estimation from
        incomplete observations. *JASA*, 53(282), 457--481.
    Kalbfleisch, J. D. & Prentice, R. L. (2002). *The Statistical
        Analysis of Failure Time Data*. 2nd ed. Wiley.
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
        in Econometrics*. Springer, Section 6.1.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if time.ndim != 1 or event.ndim != 1:
        raise ValueError("time and event must be 1-D.")
    if time.shape[0] != event.shape[0]:
        raise ValueError("time and event must have the same length.")
    n = time.shape[0]
    if n == 0:
        raise ValueError("Input arrays must not be empty.")

    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = event[order]

    unique_times = np.unique(t_sorted[e_sorted == 1])
    n_times = len(unique_times)

    surv = np.ones(n_times)
    se_arr = np.zeros(n_times)
    ci_lo = np.ones(n_times)
    ci_hi = np.ones(n_times)

    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    cum_hazard_var = 0.0
    s = 1.0

    for j, tj in enumerate(unique_times):
        n_j = np.sum(t_sorted >= tj)
        d_j = np.sum((t_sorted == tj) & (e_sorted == 1))
        if n_j > 0:
            s *= (1 - d_j / n_j)
            if n_j > d_j:
                cum_hazard_var += d_j / (n_j * (n_j - d_j))
        surv[j] = s
        var_s = s ** 2 * cum_hazard_var
        se_arr[j] = np.sqrt(max(var_s, 0))

        if s > 0 and s < 1 and se_arr[j] > 0:
            log_log_s = np.log(-np.log(s))
            se_ll = se_arr[j] / (s * abs(np.log(s)))
            ci_lo[j] = s ** np.exp(z * se_ll)
            ci_hi[j] = s ** np.exp(-z * se_ll)
        else:
            ci_lo[j] = max(s - z * se_arr[j], 0)
            ci_hi[j] = min(s + z * se_arr[j], 1)

    median_surv = np.nan
    below = np.where(surv <= 0.5)[0]
    if len(below) > 0:
        median_surv = float(unique_times[below[0]])

    return {
        "times": unique_times,
        "survival": surv,
        "se": se_arr,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "n_obs": n,
        "n_events": int(np.sum(event)),
        "median_survival": median_surv,
    }


kmsem_fn = kmsem


def cheatsheet() -> str:
    return "kmsem(time, event) -> Kaplan-Meier survival estimator."
