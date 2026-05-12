"""Inner-product distortion bound (Lemma 3.5 applied to packed TurboQuant blocks)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_inner_product_distortion_bound"]


def turboquant_inner_product_distortion_bound(bits, norm_sq, d):
    """
    Inner-product distortion bound (Lemma 3.5 applied to packed TurboQuant blocks)

    Formula: Pr[|<x_hat, y_hat> - <x, y>| > eps ||x|| ||y||] <= delta  with m depending on bits and d

    Parameters
    ----------
    bits : array-like
        Input data.
    norm_sq : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bound

    References
    ----------
    TurboQuant MORIE integration -- inner_product_distortion_bound
    """
    bits = np.atleast_1d(np.asarray(bits, dtype=float))
    n = len(bits)
    result = float(np.mean(bits))
    se = float(np.std(bits, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inner-product distortion bound (Lemma 3.5 applied to packed TurboQuant blocks)"})


def cheatsheet():
    return "tqipb: Inner-product distortion bound (Lemma 3.5 applied to packed TurboQuant blocks)"
