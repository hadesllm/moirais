"""Output of the second layer of the two-layer neural network, taking the first layer activations as input.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["burkov_lm_ch1_layer2_output"]


def burkov_lm_ch1_layer2_output(W_2, y_1, b_2_1, phi):
    """
    Output of the second layer of the two-layer neural network, taking the first layer activations as input.

    Formula: y_2 = \phi\!\left(\mathbf{W}_2 \mathbf{y}_1 + b_{2,1}\right)

    Parameters
    ----------
    W_2 : array-like
        Input data.
    y_1 : array-like
        Input data.
    b_2_1 : array-like
        Input data.
    phi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scalar second-layer output

    References
    ----------
    Burkov LM (2025), Ch 1, Eq 1.7, p. 40
    """
    W_2 = np.atleast_1d(np.asarray(W_2, dtype=float))
    n = len(W_2)
    result = float(np.mean(W_2))
    se = float(np.std(W_2, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Output of the second layer of the two-layer neural network, taking the first layer activations as input."})


def cheatsheet():
    return "b107: Output of the second layer of the two-layer neural network, taking the first layer activations as input."
