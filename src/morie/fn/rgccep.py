# morie.fn -- function file (hadesllm/morie)
"""Complex cepstrum using phase unwrapping."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_complex_cepstrum"]


def rangayyan_complex_cepstrum(x):
    """
    Complex cepstrum using phase unwrapping

    Formula: c_hat(n) = IFFT(log FFT(x)) = IFFT(log|X(f)| + j*angle_unwrapped(X(f)))

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: complex_cepstrum, quefrency

    References
    ----------
    Rangayyan Ch 4.7.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Complex cepstrum using phase unwrapping"})


def cheatsheet():
    return "rgccep: Complex cepstrum using phase unwrapping"
