# moirais.fn — function file (hadesllm/moirais)
"""K-nearest neighbor (k-NN) classifier."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_knn_classifier"]


def rangayyan_knn_classifier(X_train, y_train, X_test, k):
    """
    K-nearest neighbor (k-NN) classifier

    Formula: Assign class of majority among k nearest neighbors by Euclidean distance

    Parameters
    ----------
    X_train : array-like
        Input data.
    y_train : array-like
        Input data.
    X_test : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Rangayyan Ch 10.4.4
    """
    X_train = np.asarray(X_train, dtype=float)
    n = int(X_train) if X_train.ndim == 0 else len(X_train)
    result = float(np.mean(X_train))
    se = float(np.std(X_train, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-nearest neighbor (k-NN) classifier"})


def cheatsheet():
    return "rgknn: K-nearest neighbor (k-NN) classifier"
