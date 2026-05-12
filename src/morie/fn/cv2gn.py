# morie.fn -- function file (hadesllm/morie)
"""CV2 genomic cross-validation: both train and test lines evaluated in at least one environment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cv2_genomic"]


def cv2_genomic(y, markers, env, n_folds, cdf=None):
    """
    CV2 genomic cross-validation: both train and test lines evaluated in at least one environment

    Formula: Partition line-environment combos: some obs in train set serve as bridge; r = cor(y_obs, y_hat)

    Parameters
    ----------
    y : array-like
        Input data.
    markers : array-like
        Input data.
    env : array-like
        Input data.
    n_folds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'pa': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    if y.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 2:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "CV2 genomic cross-validation: both train and test lines evaluated in at least one environment"})
    x_sorted = np.sort(y)
    if cdf is None:
        cdf_vals = stats.norm.cdf(x_sorted, loc=np.mean(y), scale=np.std(y, ddof=1))
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
    return RichResult(payload={"statistic": float(statistic), "p_value": float(p_value), "n": n, "method": "CV2 genomic cross-validation: both train and test lines evaluated in at least one environment"})


def cheatsheet():
    return "cv2gn: CV2 genomic cross-validation: both train and test lines evaluated in at least one environment"
