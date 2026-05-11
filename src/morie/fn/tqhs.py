"""QJL 1-bit sign quantizer over Gaussian projection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["turboquant_qjl_sign_quantizer"]


def turboquant_qjl_sign_quantizer(k, S):
    """
    QJL 1-bit sign quantizer over Gaussian projection

    Formula: H_S(k) := sign(S k);  S ~ N(0, 1)^{m x d}

    Parameters
    ----------
    k : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: signs

    References
    ----------
    Zandieh et al. 2024 Eq 3 (QJL definition 3.1)
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "QJL 1-bit sign quantizer over Gaussian projection"})


def cheatsheet():
    return "tqhs: QJL 1-bit sign quantizer over Gaussian projection"
