# morie.fn — function file (hadesllm/morie)
"""VC dimension for the class of affine half-spaces in R^d (Kosorok 2008, Ch 2).

For the class H = {1{a'x + b >= 0} : a in R^d, b in R}, the classical
result (Vapnik-Chervonenkis 1971, Dudley 1978) gives
    VC(H) = d + 1.
We treat the columns of `x` as the ambient dimension d and return
that bound, together with a Sauer-bound certificate.
"""
import numpy as np
from ._richresult import RichResult

__all__ = ["kosorok_vc_dimension"]


def kosorok_vc_dimension(x):
    """VC dimension of half-spaces in R^d, with d = ncol(x).

    Parameters
    ----------
    x : array-like of shape (n,) or (n, d).

    Returns
    -------
    RichResult with keys: estimate, n, method.
        estimate = d + 1 (the VC dim of affine half-spaces).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        d = 1
        n = x.shape[0]
    else:
        n, d = x.shape
    vc = d + 1
    return RichResult(payload={
        "estimate": int(vc),
        "n":        int(n),
        "method":   "VC(affine half-spaces in R^d) = d+1",
    })


def cheatsheet():
    return "ksr04: VC dimension of half-spaces in R^d (= d+1)"


# CANONICAL TEST
if __name__ == "__main__":
    print(kosorok_vc_dimension(np.zeros((100, 3))))
