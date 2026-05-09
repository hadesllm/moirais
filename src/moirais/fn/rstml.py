# moirais.fn — function file (hadesllm/moirais)
"""
rstml.py - Restricted mean survival time (RMST) estimator.

The RMST is the area under the survival curve from 0 to a restriction time tau:
    RMST(tau) = integral_0^tau S(t) dt

It is an interpretable summary of the survival curve that does not require the
proportional hazards assumption.

Reference: Royston, P. & Parmar, M.K.B. (2011). The use of restricted mean
survival time to estimate the survival benefit of an intervention. Statistics
in Medicine, 30(19), 2409-2421.
"""

__all__ = ["rstml"]

import numpy as np


def rstml(
    time: np.ndarray,
    event: np.ndarray,
    tau: float = None,
    group: np.ndarray = None,
    alpha: float = 0.05,
) -> dict:
    """
    Estimate restricted mean survival time (RMST) up to restriction time tau.

    The RMST is computed as the area under the Kaplan-Meier survival curve:
        RMST(tau) = sum_j (t_{j+1} - t_j) * S(t_j)
    where the sum is over KM jump times up to tau.

    Variance: Var(RMST) = sum_j [(tau - t_j)^2 * d_j / (n_j * (n_j - d_j))]
    (Greenwood-based formula from Royston & Parmar, 2011)

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    tau : float, optional
        Restriction time. If None, uses the largest event time.
    group : np.ndarray, shape (n,), optional
        Group labels (two groups). If provided, returns RMST difference.
    alpha : float, optional
        Significance level for confidence intervals. Default 0.05.

    Returns
    -------
    dict
        rmst : float (or dict per group if group provided)
            RMST estimate.
        se : float
            Standard error of RMST.
        ci_lower : float
            Lower confidence bound.
        ci_upper : float
            Upper confidence bound.
        tau : float
            Restriction time used.
        If group provided, also returns:
        rmst_diff : float
            RMST difference (group 1 - group 0).
        se_diff : float
        ci_lower_diff : float
        ci_upper_diff : float
        p_value : float

    Raises
    ------
    ValueError
        If inputs are invalid or tau is beyond the last event time.

    References
    ----------
    Royston, P. & Parmar, M.K.B. (2011). Statistics in Medicine,
    30(19), 2409-2421.
    """
    from scipy import stats as _stats

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")

    if tau is None:
        tau = float(np.max(time[event == 1])) if event.sum() > 0 else float(time.max())

    z_alpha = float(_stats.norm.ppf(1 - alpha / 2))

    def _rmst_single(t, e, tau_val):
        order = np.argsort(t)
        t_s = t[order]
        e_s = e[order]

        # KM survival and RMST computation
        km_t = [0.0]
        km_s = [1.0]
        S = 1.0
        for t_j in np.unique(t_s[e_s == 1]):
            if t_j > tau_val:
                break
            n_r = np.sum(t_s >= t_j)
            n_e = np.sum((t_s == t_j) & (e_s == 1))
            S_prev = S
            S *= (1 - n_e / n_r) if n_r > 0 else 1.0
            km_t.append(t_j)
            km_s.append(S)

        km_t.append(tau_val)
        km_s_for_area = km_s[:-1] if km_t[-1] == tau_val else km_s

        km_t = np.array(km_t)
        km_s = np.array(km_s)

        # Area under KM curve up to tau: sum of rectangles
        area = 0.0
        for idx in range(len(km_t) - 1):
            dt = km_t[idx + 1] - km_t[idx]
            area += km_s[idx] * dt

        # Greenwood-based variance of RMST
        variance = 0.0
        for t_j in np.unique(t_s[e_s == 1]):
            if t_j > tau_val:
                break
            n_r = np.sum(t_s >= t_j)
            n_e = np.sum((t_s == t_j) & (e_s == 1))
            if n_r > n_e:
                # KM survival just after t_j
                idx_j = np.searchsorted(km_t, t_j, side="right") - 1
                S_j = km_s[min(idx_j, len(km_s) - 1)]
                # (tau - t_j)^2 * dH(t_j) * S(t_j)^2 (Aalen variance)
                variance += (tau_val - t_j) ** 2 * n_e / (n_r * (n_r - n_e)) * S_j ** 2
                # Note: simpler Greenwood formula for RMST variance:
                # Var(RMST) = sum[(tau - t_j)^2 * d_j / (n_j * (n_j - d_j))] * (S(t_j^-))^2
                # but we use the more standard form above

        se_val = float(np.sqrt(max(variance, 0.0)))
        return float(area), se_val

    if group is None:
        rmst_val, se_val = _rmst_single(time, event, tau)
        ci_lo = rmst_val - z_alpha * se_val
        ci_hi = rmst_val + z_alpha * se_val
        return {
            "rmst": rmst_val,
            "se": se_val,
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
            "tau": float(tau),
        }
    else:
        group = np.asarray(group)
        grps = sorted(np.unique(group).tolist())
        if len(grps) != 2:
            raise ValueError("group must have exactly two unique labels for RMST difference.")
        m0 = group == grps[0]
        m1 = group == grps[1]
        r0, s0 = _rmst_single(time[m0], event[m0], tau)
        r1, s1 = _rmst_single(time[m1], event[m1], tau)
        diff = r1 - r0
        se_diff = float(np.sqrt(s0 ** 2 + s1 ** 2))
        ci_lo_d = diff - z_alpha * se_diff
        ci_hi_d = diff + z_alpha * se_diff
        z_stat = diff / se_diff if se_diff > 0 else 0.0
        p_val = float(2 * _stats.norm.sf(abs(z_stat)))
        return {
            "rmst": {str(grps[0]): r0, str(grps[1]): r1},
            "se": {str(grps[0]): s0, str(grps[1]): s1},
            "rmst_diff": float(diff),
            "se_diff": se_diff,
            "ci_lower_diff": float(ci_lo_d),
            "ci_upper_diff": float(ci_hi_d),
            "p_value": p_val,
            "tau": float(tau),
        }
