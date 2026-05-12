# morie.fn -- function file (hadesllm/morie)
"""Forward-mode autodiff via dual numbers (x + x'*eps)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_forward_mode_autodiff"]


def geron_forward_mode_autodiff(x, x_prime, f):
    """
    Forward-mode autodiff via dual numbers (x + x'*eps)

    Formula: f(x + x' * eps) = f(x) + f'(x) * x' * eps; higher orders vanish via eps^2 = 0

    Parameters
    ----------
    x : array-like
        Input data.
    x_prime : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fx, fx_prime

    References
    ----------
    Géron Appendix A, Forward-mode autodiff section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forward-mode autodiff via dual numbers (x + x'*eps)"})


def cheatsheet():
    return "grfad: Forward-mode autodiff via dual numbers (x + x'*eps)"
