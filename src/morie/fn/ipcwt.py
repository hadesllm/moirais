# morie.fn -- function file (hadesllm/morie)
"""
ipcwt.py - Inverse probability of censoring weights (IPCW).

IPCW reweights uncensored observations to account for informative censoring,
enabling consistent estimation of marginal (covariate-unconditional) survival
quantities such as the Brier score and the Uno C-statistic.

Reference: Robins, J.M. & Rotnitzky, A. (1992). Recovery of information and
adjustment for dependent censoring using surrogate markers. In AIDS Epidemiology:
Methodological Issues. Birkhauser Boston.
"""

__all__ = ["ipcwt"]

import numpy as np


def ipcwt(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray = None,
    method: str = "km",
    tau: float = None,
) -> dict:
    """
    Compute inverse probability of censoring weights (IPCW) for each subject.

    For subject i with observed time t_i, the IPCW weight is:
        w_i = 1 / KM_C(t_i^-)
    where KM_C is the Kaplan-Meier estimator of the censoring survival function.
    Weights are set to 0 for censored subjects when used for complete-case
    weighted analysis.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times (> 0).
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    covariates : np.ndarray, optional
        Currently unused; reserved for Cox-based covariate-adjusted IPCW.
    method : str, optional
        'km': marginal KM estimate of censoring (default).
    tau : float, optional
        Restriction time. Weights are set to 0 for t_i > tau. If None, no
        restriction.

    Returns
    -------
    dict
        weights : np.ndarray, shape (n,)
            IPCW weights for each subject. Censored subjects get weight 0.
            Events get weight 1 / KM_C(t_i^-).
        km_c_times : np.ndarray
            Time points of the censoring KM estimator.
        km_c_surv : np.ndarray
            Censoring KM survival values at each time point.
        tau : float
            Restriction time used (or max event time if not specified).

    Raises
    ------
    ValueError
        If inputs are invalid.

    References
    ----------
    Robins, J.M. & Rotnitzky, A. (1992). In AIDS Epidemiology: Methodological
    Issues. Birkhauser Boston.
    Gerds, T.A. & Schumacher, M. (2006). Biometrical Journal, 48(6), 1029-1040.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    if np.any(time <= 0):
        raise ValueError("All times must be strictly positive.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")
    if method != "km":
        raise ValueError("Currently only method='km' is supported.")

    n = len(time)
    if tau is None:
        tau = float(np.max(time[event == 1])) if event.sum() > 0 else float(time.max())

    # KM estimator for censoring: treat censoring as "event", events as censored
    cen_indicator = (event == 0).astype(float)
    order = np.argsort(time)
    t_s = time[order]
    c_s = cen_indicator[order]

    cen_event_times = np.unique(t_s[c_s == 1])
    km_c_times = []
    km_c_surv = []
    S_c = 1.0

    for t_j in cen_event_times:
        n_r = np.sum(t_s >= t_j)
        n_c = np.sum((t_s == t_j) & (c_s == 1))
        km_c_times.append(t_j)
        km_c_surv.append(S_c)  # S_C(t_j^-): value just before t_j
        S_c *= (1 - n_c / n_r) if n_r > 0 else 1.0

    km_c_times = np.array(km_c_times) if km_c_times else np.array([0.0])
    km_c_surv = np.array(km_c_surv) if km_c_surv else np.array([1.0])

    def _get_km_c_before(t):
        """KM_C(t^-): value just before t."""
        idx = np.searchsorted(km_c_times, t, side="left") - 1
        if idx < 0:
            return 1.0
        idx = min(idx, len(km_c_surv) - 1)
        return float(km_c_surv[idx])

    weights = np.zeros(n)
    for i in range(n):
        if event[i] == 0:
            weights[i] = 0.0  # censored: excluded from weighted analysis
        elif time[i] > tau:
            weights[i] = 0.0  # beyond restriction time
        else:
            km_c_val = _get_km_c_before(time[i])
            weights[i] = 1.0 / max(km_c_val, 1e-10)

    return {
        "weights": weights,
        "km_c_times": km_c_times,
        "km_c_surv": km_c_surv,
        "tau": float(tau),
    }
