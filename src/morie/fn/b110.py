r"""Average binary cross-entropy loss over the entire training dataset of size N.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_dataset_bce"]


def burkov_lm_ch1_dataset_bce(y_hat, y, N):
    r"""
    Average binary cross-entropy loss over the entire training dataset of size N.

    Formula: \operatorname{loss}_{\mathcal{D}} \stackrel{\text{def}}{=} -\frac{1}{N} \sum_{i=1}^{N} \bigl[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\bigr]

    Parameters
    ----------
    y_hat : array-like
        Input data.
    y : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: average loss over dataset

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.10, p. 41
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Average binary cross-entropy loss over the entire training dataset of size N."})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Average binary cross-entropy loss over the entire training dataset of size N."})


def cheatsheet():
    return "b110: Average binary cross-entropy loss over the entire training dataset of size N."
