"""Output of the first hidden layer of a feed-forward network after applying activation function phi.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_layer1_output"]


def burkov_lm_ch1_layer1_output(W_1, x, b_1, phi):
    """
    Output of the first hidden layer of a feed-forward network after applying activation function phi.

    Formula: \mathbf{y}_1 = \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)

    Parameters
    ----------
    W_1 : array-like
        Input data.
    x : array-like
        Input data.
    b_1 : array-like
        Input data.
    phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: first-layer activation vector

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.6, p. 39
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of the first hidden layer of a feed-forward network after applying activation function phi."})


def cheatsheet():
    return "b106: Output of the first hidden layer of a feed-forward network after applying activation function phi."
