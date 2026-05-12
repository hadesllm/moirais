# morie.fn -- function file (hadesllm/morie)
"""
logrt.py - Log-rank test for comparing survival distributions.

Supports two-group and k-group comparisons. Also provides the stratified
log-rank test.

Reference: Mantel, N. (1966). Evaluation of survival data and two new rank
order statistics arising in its consideration. Cancer Chemotherapy Reports,
50(3), 163-170.
"""

__all__ = ["logrt"]

import numpy as np


def logrt(time: np.ndarray, event: np.ndarray, group: np.ndarray, rho: float = 0.0, strata: np.ndarray = None, cdf=None) -> dict:
    """
    Log-rank test (and weighted variants) for comparing survival curves
    across two or more groups.

    The test statistic is:
        Z = sum_j w_j (O_j - E_j) / sqrt(sum_j w_j^2 * V_j)
    for the two-group case, where w_j is the weight at event time t_j,
    O_j is observed and E_j is expected events in group 1.

    For k groups the chi-squared statistic is computed as
    Q = U^T V^{-1} U where U is the (k-1)-vector of observed minus expected.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    group : np.ndarray, shape (n,)
        Group labels (integer or string). Unique values determine groups.
    rho : float, optional
        Fleming-Harrington weight power rho (0 = log-rank, 1 = Wilcoxon/Breslow).
        Default 0.
    strata : np.ndarray, shape (n,), optional
        Stratum labels for stratified log-rank test. If None, unstratified.

    Returns
    -------
    dict
        statistic : float
            Chi-squared test statistic (or z-score for two groups).
        p_value : float
            p-value from chi-squared distribution with df = k-1.
        df : int
            Degrees of freedom.
        groups : list
            Ordered group labels.
        observed : np.ndarray
            Observed events per group.
        expected : np.ndarray
            Expected events per group under H0.

    Raises
    ------
    ValueError
        If fewer than 2 groups or inputs are invalid.

    References
    ----------
    Mantel, N. (1966). Cancer Chemotherapy Reports, 50(3), 163-170.
    Fleming, T.R. & Harrington, D.P. (1991). Counting Processes and Survival
    Analysis. Wiley.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    group = np.asarray(group)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")

    groups = sorted(np.unique(group).tolist())
    k = len(groups)
    if k < 2:
        raise ValueError("At least two groups required.")

    if strata is None:
        strata_labels = np.zeros(len(time), dtype=int)
    else:
        strata_labels = np.asarray(strata)

    unique_strata = np.unique(strata_labels)

    # Accumulate O - E contributions per stratum
    # U has shape (k-1,) and V has shape (k-1, k-1)
    U = np.zeros(k - 1)
    V = np.zeros((k - 1, k - 1))
    obs_total = np.zeros(k)
    exp_total = np.zeros(k)

    for s in unique_strata:
        s_mask = strata_labels == s
        t_s = time[s_mask]
        e_s = event[s_mask]
        g_s = group[s_mask]

        event_times = np.unique(t_s[e_s == 1])

        # Compute overall KM survival for weighting (rho != 0)
        S_prev = 1.0
        km_surv_map = {}
        if rho != 0:
            order = np.argsort(t_s)
            t_ord = t_s[order]
            e_ord = e_s[order]
            S = 1.0
            for t_j in event_times:
                km_surv_map[t_j] = S  # S(t_j^-)
                n_r = np.sum(t_ord >= t_j)
                n_e = np.sum((t_ord == t_j) & (e_ord == 1))
                S *= (1 - n_e / n_r) if n_r > 0 else 1.0

        for t_j in event_times:
            w_j = km_surv_map.get(t_j, 1.0) ** rho if rho != 0 else 1.0
            n_j = np.sum(t_s >= t_j)  # total at risk at t_j
            d_j = np.sum((t_s == t_j) & (e_s == 1))  # total events

            n_gk = np.array([np.sum((t_s >= t_j) & (g_s == gk)) for gk in groups])
            d_gk = np.array([np.sum((t_s == t_j) & (e_s == 1) & (g_s == gk)) for gk in groups])

            obs_total += d_gk
            if n_j <= 1:
                continue
            e_gk = n_gk * d_j / n_j  # expected under independence
            exp_total += e_gk

            # Contribution to U and V for groups 1..k-1 (group 0 is reference)
            for i in range(k - 1):
                U[i] += w_j * (d_gk[i + 1] - e_gk[i + 1])
                for j2 in range(k - 1):
                    if i == j2:
                        V[i, i] += w_j ** 2 * n_gk[i + 1] * (n_j - n_gk[i + 1]) * d_j * (n_j - d_j) / (n_j ** 2 * (n_j - 1))
                    else:
                        V[i, j2] -= w_j ** 2 * n_gk[i + 1] * n_gk[j2 + 1] * d_j * (n_j - d_j) / (n_j ** 2 * (n_j - 1))

    df = k - 1
    try:
        chi2 = float(U @ np.linalg.solve(V, U))
    except np.linalg.LinAlgError:
        chi2 = float(U @ np.linalg.lstsq(V, U, rcond=None)[0])
    chi2 = max(chi2, 0.0)
    p_val = float(1 - _stats.chi2.cdf(chi2, df=df))

    return {
        "statistic": chi2,
        "p_value": p_val,
        "df": df,
        "groups": groups,
        "observed": obs_total,
        "expected": exp_total,
    }
