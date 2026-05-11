# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""
brier.py - Brier score for survival models.

The Brier score measures the mean squared error of survival probability
predictions at a given time point, with IPCW weighting to handle censoring.

Reference: Graf, E., Schmoor, C., Sauerbrei, W. & Schumacher, M. (1999).
Assessment and comparison of prognostic classification schemes for survival
data. Statistics in Medicine, 18(17-18), 2529-2545.
"""

__all__ = ["brier"]

import numpy as np


def brier(
    time: np.ndarray,
    event: np.ndarray,
    predicted_survival: np.ndarray,
    eval_time: float,
    method: str = "ipcw",
) -> dict:
    """
    Compute the (IPCW) Brier score for a survival model at a specified time.

    The Graf et al. (1999) IPCW Brier score is:
        BS(t) = (1/n) sum_i w_i * I(T_i <= t, Delta_i = 1) * (0 - S_hat(t|x_i))^2
              + (1/n) sum_i I(T_i > t) * (1 - S_hat(t|x_i))^2
    where w_i = 1/KM_C(T_i) accounts for censoring.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Observed survival/censoring times.
    event : np.ndarray, shape (n,)
        Event indicators (1 = event, 0 = censored).
    predicted_survival : np.ndarray, shape (n,) or (n, T)
        Predicted survival probabilities S_hat(eval_time | x_i) for each subject.
        If 2D, shape (n, T) with columns corresponding to eval_time(s).
    eval_time : float or np.ndarray
        Time(s) at which to evaluate the Brier score.
    method : str, optional
        'ipcw': IPCW-weighted Brier score (default).
        'naive': unweighted (biased, for comparison only).

    Returns
    -------
    dict
        brier_score : float (or np.ndarray if multiple eval_times)
            Brier score at eval_time.
        scaled_brier : float (or np.ndarray)
            Scaled (IPA) Brier score: 1 - BS(model) / BS(null). Null is KM.
        integrated_brier : float
            Integrated Brier score (area under BS curve), if multiple eval_times.
        eval_time : float or np.ndarray
            Evaluation time(s).

    Raises
    ------
    ValueError
        If inputs have incompatible shapes or eval_time is out of range.

    References
    ----------
    Graf, E. et al. (1999). Statistics in Medicine, 18(17-18), 2529-2545.
    Gerds, T.A. & Schumacher, M. (2006). Biometrical Journal, 48(6), 1029-1040.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    predicted_survival = np.asarray(predicted_survival, dtype=float)
    n = len(time)

    if len(event) != n:
        raise ValueError("time and event must have the same length.")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be binary (0/1).")
    if method not in ("ipcw", "naive"):
        raise ValueError("method must be 'ipcw' or 'naive'.")

    scalar_eval = np.isscalar(eval_time)
    eval_times = np.atleast_1d(np.asarray(eval_time, dtype=float))
    if predicted_survival.ndim == 1:
        # Single eval time: predicted_survival is shape (n,)
        if len(eval_times) > 1:
            raise ValueError("predicted_survival must be 2D for multiple eval_times.")
        pred_2d = predicted_survival[:, None]
    else:
        pred_2d = predicted_survival
    if pred_2d.shape[0] != n:
        raise ValueError("predicted_survival must have n rows.")
    if pred_2d.shape[1] != len(eval_times):
        raise ValueError("predicted_survival columns must match number of eval_times.")

    # IPCW: KM for censoring
    cen_indicator = (event == 0).astype(float)
    order = np.argsort(time)
    t_s = time[order]
    c_s = cen_indicator[order]
    S_c = 1.0
    km_c_times = []
    km_c_vals = []
    for t_j in np.unique(t_s[c_s == 1]):
        n_r = np.sum(t_s >= t_j)
        n_c = np.sum((t_s == t_j) & (c_s == 1))
        km_c_times.append(t_j)
        km_c_vals.append(S_c)
        S_c *= (1 - n_c / n_r) if n_r > 0 else 1.0

    km_c_times = np.array(km_c_times) if km_c_times else np.array([0.0])
    km_c_vals = np.array(km_c_vals) if km_c_vals else np.array([1.0])

    def _get_km_c(t):
        idx = np.searchsorted(km_c_times, t, side="left") - 1
        if idx < 0:
            return 1.0
        return float(km_c_vals[min(idx, len(km_c_vals) - 1)])

    # KM overall survival for null model Brier score
    S_km = 1.0
    km_overall_times = []
    km_overall_vals = []
    for t_j in np.unique(t_s[c_s == 0]):  # event times
        n_r = np.sum(t_s >= t_j)
        n_e = np.sum((t_s == t_j) & (c_s == 0) & (event[order] == 1))
        if n_e == 0:
            continue
        km_overall_times.append(t_j)
        km_overall_vals.append(S_km)
        S_km *= (1 - n_e / n_r) if n_r > 0 else 1.0

    def _get_km_surv(t):
        if len(km_overall_times) == 0:
            return 1.0
        idx = np.searchsorted(km_overall_times, t, side="right") - 1
        if idx < 0:
            return 1.0
        return float(km_overall_vals[min(idx, len(km_overall_vals) - 1)])

    bs_vals = np.zeros(len(eval_times))
    bs_null_vals = np.zeros(len(eval_times))

    for k, t_eval in enumerate(eval_times):
        s_hat = pred_2d[:, k]
        s_null = _get_km_surv(t_eval)

        bs = 0.0
        bs_null = 0.0
        for i in range(n):
            if time[i] <= t_eval and event[i] == 1:
                # Subject had event before/at eval_time
                if method == "ipcw":
                    w = 1.0 / max(_get_km_c(time[i]), 1e-10)
                else:
                    w = 1.0
                bs += w * s_hat[i] ** 2
                bs_null += w * s_null ** 2
            elif time[i] > t_eval:
                # Subject still event-free at eval_time
                bs += (1.0 - s_hat[i]) ** 2
                bs_null += (1.0 - s_null) ** 2
            # Censored before eval_time: excluded (IPCW handles this implicitly)

        bs_vals[k] = bs / n
        bs_null_vals[k] = bs_null / n

    scaled_bs = np.where(bs_null_vals > 0, 1.0 - bs_vals / bs_null_vals, 0.0)

    # Integrated Brier score (trapezoidal rule) if multiple times
    if len(eval_times) > 1:
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        ibs = float(_trapz(bs_vals, eval_times)) / (eval_times[-1] - eval_times[0])
    else:
        ibs = float(bs_vals[0])

    if scalar_eval:
        return {
            "brier_score": float(bs_vals[0]),
            "scaled_brier": float(scaled_bs[0]),
            "integrated_brier": float(bs_vals[0]),
            "eval_time": float(eval_times[0]),
        }
    else:
        return {
            "brier_score": bs_vals,
            "scaled_brier": scaled_bs,
            "integrated_brier": ibs,
            "eval_time": eval_times,
        }
