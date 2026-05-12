# morie.fn -- function file (hadesllm/morie)
"""
cstat.py - Harrell's concordance statistic (C-index) for survival models.

The C-statistic measures the probability that, for a randomly chosen pair
of subjects, the one with the higher predicted risk has the shorter survival
time (among comparable pairs).

Reference: Harrell, F.E., Califf, R.M., Pryor, D.B., Lee, K.L. & Rosati, R.A.
(1982). Evaluating the yield of medical tests. JAMA, 247(18), 2543-2546.
"""

__all__ = ["cstat"]

import numpy as np


def cstat(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    method: str = "harrell",
) -> dict:
    """
    Compute the concordance statistic (C-index) for a survival model.

    A pair (i, j) is comparable (concordant/discordant) only if t_i < t_j
    and subject i experienced the event. The C-statistic is:
        C = (concordant pairs + 0.5 * tied pairs) / comparable pairs

    where concordant means risk_score_i > risk_score_j (higher risk => shorter
    survival).

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times.
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    risk_score : np.ndarray, shape (n,)
        Predicted risk scores (higher score = higher predicted risk / shorter
        expected survival). Typically exp(x^T beta) from a Cox model.
    method : str, optional
        'harrell' (default): standard Harrell C-index.
        'uno': Uno et al. (2011) IPCW-weighted C-statistic.

    Returns
    -------
    dict
        c_statistic : float
            Concordance statistic (0.5 = chance, 1.0 = perfect).
        se : float
            Standard error (Noether's variance estimator).
        ci_lower : float
            95% CI lower bound.
        ci_upper : float
            95% CI upper bound.
        concordant : int
            Number of concordant pairs.
        discordant : int
            Number of discordant pairs.
        tied : int
            Number of tied pairs.
        comparable : int
            Total number of comparable pairs.

    Raises
    ------
    ValueError
        If inputs have incompatible shapes.

    References
    ----------
    Harrell, F.E. et al. (1982). JAMA, 247(18), 2543-2546.
    Uno, H. et al. (2011). Statistics in Medicine, 30(10), 1105-1117.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    n = len(time)
    if len(event) != n or len(risk_score) != n:
        raise ValueError("time, event, and risk_score must have the same length.")
    if method not in ("harrell", "uno"):
        raise ValueError("method must be 'harrell' or 'uno'.")

    if method == "uno":
        return _uno_cstat(time, event, risk_score)

    concordant = 0
    discordant = 0
    tied = 0
    comparable = 0

    # Efficient O(n^2) computation
    for i in range(n):
        if event[i] == 0:
            continue  # censored at t_i cannot form comparable pair as subject with event
        for j in range(n):
            if i == j:
                continue
            # Pair (i, j) is comparable if t_i < t_j (i had event before j)
            if time[i] < time[j]:
                comparable += 1
                if risk_score[i] > risk_score[j]:
                    concordant += 1
                elif risk_score[i] < risk_score[j]:
                    discordant += 1
                else:
                    tied += 1
            elif time[i] == time[j] and event[j] == 1:
                # Both have events at same time: comparable pair
                comparable += 1
                if risk_score[i] > risk_score[j]:
                    concordant += 1
                elif risk_score[i] < risk_score[j]:
                    discordant += 1
                else:
                    tied += 1

    if comparable == 0:
        return {
            "c_statistic": 0.5,
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "concordant": concordant,
            "discordant": discordant,
            "tied": tied,
            "comparable": comparable,
        }

    c = (concordant + 0.5 * tied) / comparable

    # Noether (1987) variance: Var(C) = (2*C*(1-C)) / comparable
    # (conservative but simple)
    se = np.sqrt(max(2 * c * (1 - c) / comparable, 0.0))
    z = 1.959963985
    ci_lo = max(c - z * se, 0.0)
    ci_hi = min(c + z * se, 1.0)

    return {
        "c_statistic": float(c),
        "se": float(se),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
        "concordant": int(concordant),
        "discordant": int(discordant),
        "tied": int(tied),
        "comparable": int(comparable),
    }


def _uno_cstat(time, event, risk_score):
    """Uno et al. (2011) IPCW-weighted C-statistic."""
    n = len(time)
    # KM for censoring
    cen_event = (event == 0).astype(float)
    order = np.argsort(time)
    t_s = time[order]
    e_s = cen_event[order]
    S_c = 1.0
    km_c_times = []
    km_c_vals = []
    for t_j in np.unique(t_s[e_s == 1]):
        n_r = np.sum(t_s >= t_j)
        n_e = np.sum((t_s == t_j) & (e_s == 1))
        km_c_times.append(t_j)
        km_c_vals.append(S_c)
        S_c *= (1 - n_e / n_r) if n_r > 0 else 1.0

    tau = np.percentile(time[event == 1], 75) if event.sum() > 0 else time.max()

    def get_km_c(t):
        if len(km_c_times) == 0:
            return 1.0
        idx = np.searchsorted(km_c_times, t, side="right") - 1
        idx = max(idx, 0)
        return km_c_vals[min(idx, len(km_c_vals) - 1)]

    num = 0.0
    denom = 0.0
    for i in range(n):
        if event[i] == 0 or time[i] > tau:
            continue
        w_i = 1.0 / max(get_km_c(time[i]) ** 2, 1e-10)
        for j in range(n):
            if i == j:
                continue
            if time[i] < time[j]:
                denom += w_i
                if risk_score[i] > risk_score[j]:
                    num += w_i
                elif risk_score[i] == risk_score[j]:
                    num += 0.5 * w_i

    c = num / denom if denom > 0 else 0.5
    se = np.sqrt(max(2 * c * (1 - c) / max(n, 1), 0.0))
    z = 1.959963985
    return {
        "c_statistic": float(c),
        "se": float(se),
        "ci_lower": float(max(c - z * se, 0.0)),
        "ci_upper": float(min(c + z * se, 1.0)),
        "concordant": int(num),
        "discordant": int(denom - num),
        "tied": 0,
        "comparable": int(denom),
    }
