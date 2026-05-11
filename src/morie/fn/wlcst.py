"""
wlcst.py - Wilcoxon (Breslow) survival test.

This is the Gehan-Breslow-Wilcoxon test, a weighted log-rank test with
weights equal to the number at risk n_j at each event time. It gives more
weight to early differences in survival compared to the log-rank test.

Reference: Breslow, N. (1970). A generalised Kruskal-Wallis test for comparing
K samples subject to unequal patterns of censorship. Biometrika, 57(3), 579-594.
"""

__all__ = ["wlcst"]

import numpy as np


def wlcst(time: np.ndarray, event: np.ndarray, group: np.ndarray, weight: str = "breslow", cdf=None) -> dict:
    """
    Wilcoxon (Breslow) weighted log-rank test for comparing survival curves.

    The test weights each event-time contribution by w_j = n_j (number at risk)
    for the Breslow variant, or by the KM pooled survival for the Peto-Wilcoxon
    variant.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    group : np.ndarray, shape (n,)
        Group labels. Currently supports two groups.
    weight : str, optional
        Weighting scheme: 'breslow' (n_j at risk, default) or 'peto' (KM estimate).

    Returns
    -------
    dict
        statistic : float
            Chi-squared statistic (1 df).
        p_value : float
        z_score : float
            Signed z-score (positive = group 1 has worse survival).
        observed : np.ndarray, shape (2,)
            Observed events per group.
        expected : np.ndarray, shape (2,)
            Expected events per group under H0.

    Raises
    ------
    ValueError
        If not exactly two groups or inputs are invalid.

    References
    ----------
    Breslow, N. (1970). Biometrika, 57(3), 579-594.
    Peto, R. & Peto, J. (1972). Journal of the Royal Statistical Society,
    Series A, 135(2), 185-207.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)

    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")
    if weight not in ("breslow", "peto"):
        raise ValueError("weight must be 'breslow' or 'peto'.")

    groups = sorted(np.unique(group).tolist())
    if len(groups) != 2:
        raise ValueError("wlcst currently supports exactly two groups.")

    g1_mask = group == groups[0]
    g2_mask = group == groups[1]
    event_times = np.unique(time[event == 1])

    # KM pooled survival for Peto weights
    S_peto = 1.0
    km_map = {}
    if weight == "peto":
        order = np.argsort(time)
        t_ord = time[order]
        e_ord = event[order]
        S = 1.0
        for t_j in event_times:
            km_map[t_j] = S
            n_r = np.sum(t_ord >= t_j)
            n_e = np.sum((t_ord == t_j) & (e_ord == 1))
            S *= (1 - n_e / n_r) if n_r > 0 else 1.0

    numerator = 0.0
    variance = 0.0
    obs_1 = 0.0
    exp_1 = 0.0
    obs_2 = 0.0
    exp_2 = 0.0

    for t_j in event_times:
        n_j = np.sum(time >= t_j)
        n_1j = np.sum((time >= t_j) & g1_mask)
        n_2j = np.sum((time >= t_j) & g2_mask)
        d_j = np.sum((time == t_j) & (event == 1))
        d_1j = np.sum((time == t_j) & (event == 1) & g1_mask)
        d_2j = np.sum((time == t_j) & (event == 1) & g2_mask)

        obs_1 += d_1j
        obs_2 += d_2j

        if n_j <= 1:
            continue

        e_1j = n_1j * d_j / n_j
        exp_1 += e_1j
        exp_2 += n_2j * d_j / n_j

        if weight == "breslow":
            w_j = n_j
        else:  # peto
            w_j = km_map.get(t_j, 1.0)

        numerator += w_j * (d_1j - e_1j)
        variance += w_j ** 2 * n_1j * n_2j * d_j * (n_j - d_j) / (n_j ** 2 * (n_j - 1))

    if variance <= 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "z_score": 0.0,
            "observed": np.array([obs_1, obs_2]),
            "expected": np.array([exp_1, exp_2]),
        }

    z = numerator / np.sqrt(variance)
    chi2 = z ** 2
    p_val = float(1 - _stats.chi2.cdf(chi2, df=1))

    return {
        "statistic": float(chi2),
        "p_value": p_val,
        "z_score": float(z),
        "observed": np.array([obs_1, obs_2]),
        "expected": np.array([exp_1, exp_2]),
    }
