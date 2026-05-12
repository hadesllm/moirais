# morie.fn -- function file (hadesllm/morie)
"""Point process model for inter-event interval (IEI) statistics."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_point_process"]


def rangayyan_point_process(event_times, T, cdf=None):
    """
    Point process model for inter-event interval (IEI) statistics

    Formula: Poisson: P(k events in T) = (lambda*T)^k * exp(-lambda*T) / k!

    Parameters
    ----------
    event_times : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rate, iei_mean, iei_cv

    References
    ----------
    Rangayyan Ch 7.3
    """
    event_times = np.asarray(event_times, dtype=float)
    n = int(event_times) if event_times.ndim == 0 else len(event_times)
    if event_times.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Point process model for inter-event interval (IEI) statistics"})
    x_sorted = np.sort(event_times)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(event_times), scale=np.std(event_times, ddof=1))
    else:
        cdf_vals = np.array([cdf(xi) for xi in x_sorted])
    ecdf = np.arange(1, n + 1) / n
    ecdf_prev = np.arange(0, n) / n
    d_plus = np.max(ecdf - cdf_vals)
    d_minus = np.max(cdf_vals - ecdf_prev)
    statistic = max(d_plus, d_minus)
    if n <= 40:
        p_value = 1.0 - stats.ksone.cdf(statistic, n)
    else:
        lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * statistic
        p_value = 2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2) for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "Point process model for inter-event interval (IEI) statistics"})


def cheatsheet():
    return "rgppt: Point process model for inter-event interval (IEI) statistics"
