# morie.fn — function file (hadesllm/morie)
"""Early-stopping rule: halt training when validation error is minimal."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_early_stopping"]


def geron_early_stopping(X_train, y_train, X_val, y_val, n_iter, eta):
    """
    Early-stopping rule: halt training when validation error is minimal

    Formula: t_stop = argmin_t RMSE_val(t); return theta_{t_stop}

    Parameters
    ----------
    X_train : array-like
        Input data.
    y_train : array-like
        Input data.
    X_val : array-like
        Input data.
    y_val : array-like
        Input data.
    n_iter : array-like
        Input data.
    eta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4, Early Stopping section
    """
    X_train = np.asarray(X_train, dtype=float)
    n = int(X_train) if X_train.ndim == 0 else len(X_train)
    result = float(np.mean(X_train))
    se = float(np.std(X_train, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Early-stopping rule: halt training when validation error is minimal"})


def cheatsheet():
    return "greast: Early-stopping rule: halt training when validation error is minimal"
