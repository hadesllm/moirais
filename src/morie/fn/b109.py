r"""Binary cross-entropy (logistic) loss for a single example in binary classification.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_binary_cross_entropy"]


def burkov_lm_ch1_binary_cross_entropy(y_hat_i, y_i):
    r"""
    Binary cross-entropy (logistic) loss for a single example in binary classification.

    Formula: \operatorname{loss}(\hat{y}_i, y_i) \stackrel{\text{def}}{=} -\bigl[y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\bigr]

    Parameters
    ----------
    y_hat_i : array-like
        Input data.
    y_i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: non-negative loss

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.9, p. 40
    """
    y_hat_i = np.atleast_1d(np.asarray(y_hat_i, dtype=float))
    n = len(y_hat_i)
    result = float(np.mean(y_hat_i))
    se = float(np.std(y_hat_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary cross-entropy (logistic) loss for a single example in binary classification."})


def cheatsheet():
    return "b109: Binary cross-entropy (logistic) loss for a single example in binary classification."
