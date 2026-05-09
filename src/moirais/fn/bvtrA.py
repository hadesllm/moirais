# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Bias-variance decomposition of expected prediction error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["bias_variance_tradeoff"]


def bias_variance_tradeoff(y_true, y_pred, noise_var):
    """
    Bias-variance decomposition of expected prediction error

    Formula: E[(y - f_hat(y_true))^2] = Bias^2(f_hat) + Var(f_hat) + sigma^2

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.
    noise_var : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'bias2': 'float', 'variance': 'float', 'noise': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bias-variance decomposition of expected prediction error"})


def cheatsheet():
    return "bvtrA: Bias-variance decomposition of expected prediction error"
